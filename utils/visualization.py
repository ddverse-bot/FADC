```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(
        self,
        save_dir="results"
    ):

        self.save_dir = save_dir
        os.makedirs(
            save_dir,
            exist_ok=True
        )

    ##################################################
    # Draw GT and Prediction contours
    ##################################################

    def draw_segmentation(
        self,
        image,
        gt_mask,
        pred_mask,
        dice_score=None,
        save_name="segmentation.png"
    ):

        image = image.copy()

        if image.ndim == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        image = image.astype(
            np.uint8
        )

        gt = (
            gt_mask > 0.5
        ).astype(
            np.uint8
        )

        pred = (
            pred_mask > 0.5
        ).astype(
            np.uint8
        )

        gt_contours, _ = cv2.findContours(
            gt,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        pred_contours, _ = cv2.findContours(
            pred,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        vis = image.copy()

        # GT = green
        cv2.drawContours(
            vis,
            gt_contours,
            -1,
            (0, 255, 0),
            2
        )

        # Prediction = red
        cv2.drawContours(
            vis,
            pred_contours,
            -1,
            (255, 0, 0),
            2
        )

        if dice_score is not None:

            cv2.putText(
                vis,
                f"{dice_score:.4f}",
                (10, vis.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imwrite(
            os.path.join(
                self.save_dir,
                save_name
            ),
            vis
        )

        return vis

    ##################################################
    # Uncertainty heatmap
    ##################################################

    def draw_uncertainty(
        self,
        uncertainty_map,
        save_name="uncertainty.png"
    ):

        unc = uncertainty_map.copy()

        unc = np.clip(
            unc,
            0.1,
            0.5
        )

        normalized = (
            unc - 0.1
        ) / (
            0.5 - 0.1
        )

        normalized = np.clip(
            normalized,
            0,
            1
        )

        heatmap = plt.cm.jet(
            normalized
        )[:, :, :3]

        heatmap = (
            heatmap * 255
        ).astype(
            np.uint8
        )

        # make low uncertainty background white
        background = (
            unc <= 0.1
        )

        heatmap[
            background
        ] = np.array(
            [255, 255, 255]
        )

        cv2.imwrite(
            os.path.join(
                self.save_dir,
                save_name
            ),
            cv2.cvtColor(
                heatmap,
                cv2.COLOR_RGB2BGR
            )
        )

        return heatmap

    ##################################################
    # Complete figure
    ##################################################

    def save_complete_visualization(
        self,
        image,
        gt,
        pred,
        uncertainty,
        dice,
        idx
    ):

        self.draw_segmentation(
            image,
            gt,
            pred,
            dice_score=dice,
            save_name=
            f"{idx}_seg.png"
        )

        self.draw_uncertainty(
            uncertainty,
            save_name=
            f"{idx}_unc.png"
        )
```
