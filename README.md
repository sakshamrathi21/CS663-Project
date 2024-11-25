# CS663-Project

## Team Members
1. Saksham Rathi (22B1003)
2. Kavya Gupta (22B1053)
3. Shravan Srinivasa Raghavan (22B1054)

## Project Description
For the basic part of this project, we implemented an image compression algorithm similar to JPEG for grayscale images (using Discrete Cosine Transform (DCT) and Quantization).

In the advanced part of this project, we:-
- integrated further compression using Runlength encoding and zigzag ordering,
- modified the basic algorithm for compressing colour (RGB) images,
- implemented a PCA-based image compression algorithm for single image (grayscale and colour) and dataset of grayscale images, and,
- researched, implemented and innovated on the paper "Edge-Based Image Compression with Homogeneous Diffusion".

## How to download the datasets
The folders containing the datasets are available at:
- https://github.com/sakshamrathi21/Image-Compression-Project/tree/main/images OR
- https://drive.google.com/drive/folders/14ArHoenKSdaGJ7WB250A8ZkXXRmb84wY?usp=sharing

You can download the datasets from the above link. Make sure to place the datasets in the `images` directory.

## Instructions to run the code
- Clone the repository using the following command:
`git clone https://github.com/sakshamrathi21/Image-Compression-Project.git`
- Navigate to the `main` directory in the cloned repository.

### Basic Part
- To run the basic image compression algorithm (code in `main/basic.py`), run the following command:
`python3 run.py basic`

### Advanced Part
- To run basic code modified with runlength encoding and zigzag ordering (code in `main/runlength.py`), run the following command:
`python3 run.py runlength`
- To run the code for compressing colour images (code in `main/colour.py`), run the following command:
`python3 run.py colour`
- To run the PCA-based image compression algorithm for single grayscale image (code in `main/pca.py`), run the following command:
`python3 run.py pca`
- To run the PCA-based image compression algorithm for single colour image (code in `main/pca.py`), run the following command:
`python3 run.py cpca`
- To run the PCA-based image compression algorithm for dataset of grayscale images (code in `main/pca.py`), run the following command:
`python3 run.py dpca`
- To run the code for "Edge-Based Image Compression with Homogeneous Diffusion" (code in `main/paper.py`), run the following command:
`python3 run.py paper`

## Downloading JBIG-KIT
It is needed for the paper implementation. You can download it from the following link:
`git clone https://github.com/Distrotech/jbigkit.git`

## Report
The report for the project can be found at `documentation/presentation.pdf`.

## Notes
- `algorithms` directory contains helper files.
- `images` directory contains images (datasets) used for testing.
- You will see the output images (if any) in the `results` directory.
- `include` directory contains common Python imports.
- `jbigkit` directory contains the JBIG-KIT library for the paper implementation.