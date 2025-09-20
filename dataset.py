import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
from collections import defaultdict
import constants




class HistopathologyContrastiveDataset(Dataset):
    def __init__(self, img_dir, caption_dir, tokenizer_name="bert-base-uncased", transform=None):
        """
        Args:
            img_dir (str): Path to the image folder.
            caption_dir (str): Path to the captions folder.
            tokenizer_name (str): Name of the tokenizer (e.g., BERT).
            transform (callable, optional): Transform to be applied to the images.
        """
        self.img_dir = img_dir
        self.caption_dir = caption_dir
        self.classes = ["Alveoli", "Necrosis", "Immune Cells", "Other", "Tumor", "Background", "Stroma"]
        self.biomarkers = ["P53", "Ki67", "cyclin-D1", "CDK4", "CD38", "CD68", "CD34", "CD3", "SMA", "D2-40", "CD20", "FAP"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mean = constants.mean
        self.std = constants.std

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=constants.mean,std=constants.std)
        ])

        # Generate class-to-index mapping
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.combination_to_index = {
            f"{cls}_{biom}": idx
            for idx, (cls, biom) in enumerate(
                [(cls, biom) for cls in self.classes for biom in self.biomarkers]
            )
        }

        self.data = self._load_data()

    def _load_data(self):
        """Parse the dataset and prepare image-caption-label triplets."""
        data = []
        for class_name in self.classes:
            class_path = os.path.join(self.img_dir, class_name)
            caption_class_path = os.path.join(self.caption_dir, class_name)

            if not os.path.exists(class_path) or not os.path.exists(caption_class_path):
                print(f"Skipping class '{class_name}': missing image or caption folder")
                continue

            for img_idx, img_name in enumerate(os.listdir(class_path)):
                # Skip invalid image files
                if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # Extract biomarker
                biomarker = self._extract_biomarker(img_name)
                if not biomarker:
                    continue

                # Find corresponding caption file
                caption_file = os.path.join(caption_class_path, f"{biomarker}.txt")
                if not os.path.exists(caption_file):
                    continue

                # Read captions
                with open(caption_file, "r") as f:
                    captions = f.readlines()

                if not captions:  # Skip if no captions are available
                    print(f"Skipping file '{img_name}': no captions available in '{caption_file}'")
                    continue

                # Pair each image with exactly one caption (cycle captions if fewer)
                caption = captions[img_idx % len(captions)].strip()

                # Create labels
                label = torch.zeros(len(self.classes) + len(self.combination_to_index))
                label[self.class_to_index[class_name]] = 1
                combination_key = f"{class_name}_{biomarker}"
                if combination_key in self.combination_to_index:
                    label[len(self.classes) + self.combination_to_index[combination_key]] = 1

                # Add to data
                data.append({
                    "image_path": os.path.join(class_path, img_name),
                    "caption": caption,
                    "img_label": label,
                    "text_label": label
                })

        print(f"Total images loaded: {len(data)}")
        return data


    def _extract_biomarker(self, filename):
        """Extract biomarker from the filename."""
        for biomarker in self.biomarkers:
            if biomarker.lower() in filename.lower():
                return biomarker
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Process the image
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenize the caption
        caption_tokens = self.tokenizer(
            item["caption"],
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": caption_tokens["input_ids"].squeeze(0),
            "attention_mask": caption_tokens["attention_mask"].squeeze(0),
            "img_label": item["img_label"],
            "text_label": item["text_label"],
            
        }

class HistopathologyContrastiveCollator:
    def __init__(self, tokenizer_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, batch):
        inputs = defaultdict(list)

        # Collect inputs
        for data in batch:
            inputs["pixel_values"].append(data["pixel_values"])
            inputs["input_ids"].append(data["input_ids"])
            inputs["attention_mask"].append(data["attention_mask"])
            inputs["img_labels"].append(data["img_label"])
            inputs["text_labels"].append(data["text_label"])
            

        # Stack inputs
        inputs["pixel_values"] = torch.stack(inputs["pixel_values"])
        inputs["input_ids"] = torch.stack(inputs["input_ids"])
        inputs["attention_mask"] = torch.stack(inputs["attention_mask"])
        inputs["img_labels"] = torch.stack(inputs["img_labels"])
        inputs["text_labels"] = torch.stack(inputs["text_labels"])

        return inputs



class ZeroShotImageDataset(Dataset):
    def __init__(self, img_dir, class_names, prompts_per_class, img_transform=None):
        """
        Args:
            img_dir (str): Path to the image folder.
            class_names (list): List of class names (folder names).
            prompts_per_class (dict): Dictionary of class names and their associated prompts.
            img_transform (callable, optional): Transformations for images.
        """
        self.img_dir = img_dir
        self.class_names = class_names
        self.prompts_per_class = prompts_per_class  # Captions grouped by class
        self.mean = constants.mean
        self.std =constants.std
        self.transform = img_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.data = self._load_data()

    def _load_data(self):
        """Loads images and their labels based on folder structure."""
        data = []
        for class_name in self.class_names:
            class_folder = os.path.join(self.img_dir, class_name)
            if not os.path.exists(class_folder):
                print(f"Skipping missing folder: {class_folder}")
                continue

            for img_name in os.listdir(class_folder):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        "image_path": os.path.join(class_folder, img_name),
                        "label": class_name,  # Folder name as label
                    })

        print(f"Loaded {len(data)} images.")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = item["label"]
        return image, label

class ZeroShotImageCollator:
    def __init__(self, tokenizer_name="bert-base-uncased", prompts_per_class=None, mode="multiclass"):
        """
        Args:
            tokenizer_name (str): Name of the tokenizer to use.
            prompts_per_class (dict): Dictionary of class names and their associated prompts.
            mode (str): Evaluation mode ('multiclass', 'multilabel', or 'binary').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.prompts_per_class = prompts_per_class
        self.mode = mode

    def __call__(self, batch):
        inputs = defaultdict(list)
        prompt_inputs = defaultdict(list)

        for image, label in batch:
            inputs['pixel_values'].append(image)
            inputs['labels'].append(label)

        # Tokenize prompts
        for class_name, prompts in self.prompts_per_class.items():
            tokenized_prompts = self.tokenizer(
                prompts,
                truncation=True,
                padding=True,
                max_length=77,
                return_tensors="pt"
            )
            prompt_inputs[class_name] = tokenized_prompts

        # Stack images
        inputs['pixel_values'] = torch.stack(inputs['pixel_values'])
        inputs['prompt_inputs'] = dict(prompt_inputs)


        # Process labels for evaluation mode
        if self.mode == "multiclass":
            class_to_idx = {cls: idx for idx, cls in enumerate(self.prompts_per_class.keys())}
            inputs['labels'] = torch.tensor([class_to_idx[label] for label in inputs['labels']], dtype=torch.long)
            print(f"Pixel Values Shape: {inputs['pixel_values'].shape}")
            print(f"Labels Shape: {inputs['labels'].shape}, Unique Labels: {torch.unique(inputs['labels'])}")

        elif self.mode == "multilabel":
            pass  # Handle multilabel logic if needed
        else:
            raise ValueError("Invalid mode. Choose 'multiclass' or 'multilabel'.")

        return inputs
