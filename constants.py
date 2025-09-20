#10k
#mean = [0.7628336548805237, 0.7380197048187256, 0.7423542737960815]
#std = [0.22081395983695984, 0.22079642117023468, 0.21473398804664612]
#50k
mean= [0.7424656748771667, 0.7176911234855652, 0.7305417060852051]
std =[0.21424046158790588, 0.21373531222343445, 0.20845921337604523]
cls_prompts = {
    "Immune Cells": [
        "An IHC stained image of Immune Cells",
        "An IHC image of Immune cell stained using the cyclin-D1 biomarker.",
        "An IHC image of Immune cell stained using the CD20 biomarker.",
        "An IHC image of Immune cell stained using the CD34 biomarker",
        "An IHC image of Immune cell stained using the CD38 biomarker.",
        "An IHC image of Immune cell stained using the CD68 biomarker",
        "An IHC image of Immune cell stained using the CDK4 biomarker",
        "An IHC image of Immune cell stained using the CD3 biomarker.,",
        "An IHC image of Immune cell stained using the D2 biomarker.",
        "An IHC image of Immune cell stained using the FAP biomarker",
        "An IHC image of Immune cell stained using the CKi67 biomarker",
        "An IHC image of Immune cell stained using the P53 biomarker.",
        "An IHC image of Immune cell stained using the SMA biomarker"
    ],
    "Alveoli": [
        "An IHC stained image of Alveoli",
        "An IHC image of Alveoli stained using the cyclin-D1 biomarker.",
        "An IHC image of Alveoli stained using the CD34 biomarker",
        "An IHC image of Alveoli stained using the CD38 biomarker.",
        "An IHC image of Alveoli stained using the CD68 biomarker",
        "An IHC image of Alveoli stained using the CDK4 biomarker",
        "An IHC image of Alveoli stained using the CD3 biomarker.,",
        "An IHC image of Alveoli stained using the D2 biomarker.",
        "An IHC image of Alveoli stained using the FAP biomarker",
        "An IHC image of Alveoli stained using the CKi67 biomarker",
        "An IHC image of Alveoli stained using the P53 biomarker.",
        "An IHC image of Alveoli stained using the SMA biomarker"
    ],
    "Necrosis": [
        "An IHC stained image of Necrosis",
        "An IHC image of Necrosis stained using the cyclin-D1 biomarker.",
        "An IHC image of Necrosis stained using the CD34 biomarker",
        "An IHC image of Immune cell stained using the CD20 biomarker."
        "An IHC image of Necrosis stained using the CD38 biomarker.",
        "An IHC image of Necrosis stained using the CD68 biomarker",
        "An IHC image of Necrosis stained using the CDK4 biomarker",
        "An IHC image of Necrosis stained using the CD3 biomarker.,",
        "An IHC image of Necrosis stained using the D2 biomarker.",
        "An IHC image of Necrosis stained using the FAP biomarker",
        "An IHC image of Necrosis stained using the CKi67 biomarker",
        "An IHC image of Necrosis stained using the P53 biomarker.",
        "An IHC image of Necrosis stained using the SMA biomarker"
    ],
    "Stroma": [
        "An IHC stained image of Stroma",
        "An IHC image of Stroma stained using the cyclin-D1 biomarker.",
        "An IHC image of Stroma stained using the CD34 biomarker",
        "An IHC image of Immune cell stained using the CD20 biomarker.",
        "An IHC image of Stroma stained using the CD38 biomarker.",
        "An IHC image of Stroma stained using the CD68 biomarker",
        "An IHC image of Stroma stained using the CDK4 biomarker",
        "An IHC image of Stroma stained using the CD3 biomarker.,",
        "An IHC image of Stroma stained using the D2 biomarker.",
        "An IHC image of Stroma stained using the FAP biomarker",
        "An IHC image of Stroma stained using the CKi67 biomarker",
        "An IHC image of Stroma stained using the P53 biomarker.",
        "An IHC image of Stroma stained using the SMA biomarker"
    ],
    "Background": [
        "An IHC stained image of Background",
        "An IHC image of Background stained using the CKi67 biomarker",
        "An IHC image of Background stained using the CD3 biomarker"
    ],
    "Other": [
        "An IHC stained image of Other",
        "An IHC image of Other stained using the cyclin-D1 biomarker.",
        "An IHC image of Other stained using the CD34 biomarker",
        "An IHC image of Other stained using the CD38 biomarker.",
        "An IHC image of Immune cell stained using the CD20 biomarker.",
        "An IHC image of Other stained using the CD68 biomarker",
        "An IHC image of Other stained using the CDK4 biomarker",
        "An IHC image of Other stained using the CD3 biomarker.,",
        "An IHC image of Other stained using the D2 biomarker.",
        "An IHC image of Other stained using the FAP biomarker",
        "An IHC image of Other stained using the CKi67 biomarker",
        "An IHC image of Other stained using the P53 biomarker.",
        "An IHC image of Other stained using the SMA biomarker"
    ],
    "Tumor": [
        "An IHC stained image of Tumor",
        "An IHC image of Tumor stained using the cyclin-D1 biomarker.",
        "An IHC image of Tumor stained using the CD34 biomarker",
        "An IHC image of Tumor stained using the CD38 biomarker.",
        "An IHC image of Immune cell stained using the CD20 biomarker.",
        "An IHC image of Tumor stained using the CD68 biomarker",
        "An IHC image of Tumor stained using the CDK4 biomarker",
        "An IHC image of Tumor stained using the CD3 biomarker.,",
        "An IHC image of Tumor stained using the D2 biomarker.",
        "An IHC image of Tumor stained using the FAP biomarker",
        "An IHC image of Tumor stained using the CKi67 biomarker",
        "An IHC image of Tumor stained using the P53 biomarker.",
        "An IHC image of Tumor stained using the SMA biomarker"
    ]
}
Pantumor_cls_prompts = {
    "Tumor cell": [
        "An IHC-stained image of a tumor",
    ],
    "CD3+ immune cell": [
        "An IHC-stained image of immune cell tissue stained with CD3+",
    ],
    "Other cell": [
        "An IHC-stained image of other tissue",
    ]
}

Bc_cls_prompts = {
    "Tumor": [
        "An IHC-stained image of a tumor",
        "An IHC-stained image using Ki-67 of a tumor"
    ],
    "Non Tumor": [
        "An IHC-stained image of non-tumor tissue"
    ]
}


HNSCC_cls_prompts = {
    "T": [
        "An IHC-stained image of the tumor core ",
    ],
    "M": [
        "An IHC-stained image of the tumor margin",
    ],
    "S": [
        "An IHC-stained image of the stroma tissue",
    ]
}
Mist_cls_prompts = {
    "ER": [
        "An IHC-stained histology image showing estrogen receptor (ER) expression.",
        "Microscopic tissue sample stained for ER biomarker using immunohistochemistry.",
        "A breast cancer slide with positive ER staining.",
        "Histopathology slide of ER-positive tumor cells.",
        "A tissue image showing nuclear expression of estrogen receptor."
    ],
    "PR": [
        "An IHC-stained histology image showing progesterone receptor (PR) expression.",
        "Microscopic tissue stained for PR using immunohistochemistry.",
        "A breast cancer slide with positive PR staining.",
        "Histopathology image with PR-positive cells.",
        "A slide showing nuclear expression of the progesterone receptor biomarker."
    ],
    "HER2": [
        "An IHC image showing HER2 protein overexpression in cancer tissue.",
        "Histology slide stained for HER2 biomarker using immunohistochemistry.",
        "A breast cancer sample showing HER2-positive staining pattern.",
        "Microscopic image highlighting membrane expression of HER2.",
        "Tissue image with strong membranous HER2 staining."
    ],
    "KI67": [
        "An IHC-stained histology image showing Ki67 proliferation marker expression.",
        "A slide showing Ki67-positive nuclei in cancer tissue.",
        "Microscopic image of a tumor with Ki67 staining.",
        "Histopathology sample stained for Ki67 using immunohistochemistry.",
        "High Ki67 expression indicating cellular proliferation in tumor cells."
    ]
}
Acrobat_cls_prompts = {
    "ER": [
        "An IHC-stained histology image showing estrogen receptor (ER) expression.",
        "Microscopic tissue sample stained for ER biomarker using immunohistochemistry.",
        "A breast cancer slide with positive ER staining.",
        "Histopathology slide of ER-positive tumor cells.",
        "A tissue image showing nuclear expression of estrogen receptor."
    ],
    "KI67": [
        "An IHC-stained histology image showing Ki67-positive proliferating cells.",
        "Microscopic tumor sample stained for Ki67 proliferation marker using immunohistochemistry.",
        "Histopathology slide highlighting nuclear Ki67 expression in dividing cells.",
        "A tissue section showing high Ki67 labeling index in tumor cells.",
        "IHC of tissue indicating active cell proliferation through Ki67 staining."
    ],
    "HER2": [
        "An IHC image showing HER2 protein overexpression in cancer tissue.",
        "Histology slide stained for HER2 biomarker using immunohistochemistry.",
        "A breast cancer sample showing HER2-positive staining pattern.",
        "Microscopic image highlighting membrane expression of HER2.",
        "Tissue image with strong membranous HER2 staining."
    ],
    "PGR": [
        "An IHC-stained histology image showing progesterone receptor (PgR) expression.",
        "Microscopic breast tissue section stained for PgR biomarker using immunohistochemistry.",
        "A breast cancer slide with positive PgR staining in tumor cells.",
        "Histopathology image highlighting nuclear PgR expression.",
        "A tissue image showing progesterone receptor positivity in cancer cells."
    ]
}
Anhir_cls_prompts = {
    "CD31": [
        "An IHC-stained histology image showing CD31-positive endothelial cells lining blood vessels.",
        "Microscopic tissue section stained for CD31 biomarker using immunohistochemistry.",
        "Histopathology slide highlighting CD31 expression in vascular endothelial cells.",
        "A tissue image with strong membranous CD31 staining in capillaries and small vessels.",
        "IHC of tumor microenvironment showing CD31-positive vascular structures."
    ],
    "Ki67": [
        "An IHC-stained histology image showing Ki67-positive proliferating cells.",
        "Microscopic tumor sample stained for Ki67 proliferation marker using immunohistochemistry.",
        "Histopathology slide highlighting nuclear Ki67 expression in dividing cells.",
        "A tissue section showing high Ki67 labeling index in tumor cells.",
        "IHC of tissue indicating active cell proliferation through Ki67 staining."
    ],
    "proSPC": [
        "An IHC-stained histology image showing pro–surfactant protein C (proSPC) expression in alveolar type II cells.",
        "Microscopic lung tissue section stained for proSPC biomarker using immunohistochemistry.",
        "Histopathology slide demonstrating cytoplasmic proSPC expression in alveolar epithelium.",
        "A lung tissue image with proSPC-positive type II pneumocytes.",
        "IHC of pulmonary tissue showing proSPC expression in surfactant-producing cells."
    ],
}
her2_cls_prompts = {
    0: [
        "An IHC stained image of healthy tissue.",
        "An IHC stained image showing normal tissue.",
        "An IHC stained image of non-tumor tissue."
    ],
    1: [
        "An IHC stained image of tumor tissue.",
        "An immunohistochemically stained image showcasing tumor tissue.",
        "The image presents tumor tissue visualized through IHC staining."
    ]
}

hc4bc_cls_promptss = {
    "cls_1.0": [
        "The image shows a positive staining",
    ],
    "cls_0.0": [
        "The image shows a negative staining",
    ],
}

BC_cls_promptss = {
    "cls_1.0": [
        "The image shows a positive staining",
    ],
    "cls_0.0": [
        "The image shows a negative staining",
    ],
}