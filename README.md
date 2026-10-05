# Semi-Autonomous Aerial Surveillance System

Computer-vision pipeline for a **drone-based surveillance system** that detects changes on the ground by comparing live aerial footage against a reference map — and marks where those changes are on the map. Built as a final-year engineering design project, including a **GPU-accelerated** version using OpenCL.

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![OpenCL](https://img.shields.io/badge/OpenCL-GPU-red)

## How it works
```
 drone video ──► sample every 10th frame
                       │
 reference map ──► 1. locate frame on map        SIFT features + matching
                       │
                   2. align (perspective warp)    homography / linear alignment
                       │
                   3. difference                  pixel-wise threshold   (OpenCL kernel)
                       │
                   4. clean up noise              erosion / dilation     (OpenCL kernel)
                       │
                   5. detect & draw changes       contours, bounding boxes
                       │
                   6. project onto the map        rotation by estimated drone heading
```
1. **Localisation** – SIFT keypoints from the current frame are matched against the reference map to find the corresponding map region.
2. **Registration** – the map patch is warped into the frame's perspective so both images line up pixel-for-pixel.
3. **Change detection** – a thresholded absolute difference highlights pixels that changed (threshold set from the GUI slider).
4. **Morphological filtering** – repeated erosion/dilation removes noise; contours that are too small or too large are discarded.
5. **Reporting** – detected changes are boxed on the frame and their positions, rotated by the drone's heading, are drawn on the map.

## Repository layout
| Folder | Description |
|---|---|
| [`src/video-pipeline-gpu`](src/video-pipeline-gpu) | **Main pipeline.** Video input; SIFT, differencing and morphology run on the GPU via [silx](https://www.silx.org/) and PyOpenCL kernels (`*.cl`). Entry point: `Surveillance_System.py`. |
| [`src/video-pipeline-cpu`](src/video-pipeline-cpu) | CPU version for video input using OpenCV SIFT + FLANN. Entry point: `ProjectGUI.py`. |
| [`src/image-pipeline`](src/image-pipeline) | CPU version that processes a folder of still images. Entry point: `ProjectGUI.py`. |

## Running
```bash
pip install -r requirements.txt
cd src/video-pipeline-gpu          # kernels are loaded relative to this folder
python Surveillance_System.py
```
In the Tkinter GUI, choose the drone video (`.mp4`) and the reference map image (`.jpg`), set the difference threshold and press **Process**. Annotated frames and the change map are written to `Result_Images/`.

The GPU pipeline needs an OpenCL-capable GPU and drivers. The CPU pipelines additionally expect pre-computed SIFT keypoints/descriptors for each map tile (`kp<N>.txt`, `des<N>.txt`) in a folder named by the `FEATURES_DIR` environment variable (default `./features`).

## Possible improvements
- Replace the hand-written match filtering with RANSAC (`cv2.findHomography(..., cv2.RANSAC)`) for more robust alignment.
- Share one code base between the three pipelines (they duplicate most modules) with a CPU/GPU backend switch.
- Add a script to generate the map feature cache, and a small sample video + map so the demo runs out of the box.
- Swap pixel differencing for a learned change-detection model to reduce false positives from lighting and shadows.
