"""
Grad-CAM — Fixed for MobileNetV2
Uses 'Conv_1' (last feature conv before GAP) for accurate crack heatmaps.
"""

import numpy as np
import cv2
import tensorflow as tf


class GradCAM:
    def __init__(self, model):
        self.model      = model
        self.layer_name = self._find_best_layer()
        print(f"  ✅ Grad-CAM layer: {self.layer_name}")

    def _find_best_layer(self):
        """
        For MobileNetV2, 'Conv_1' is the best layer — it's the last
        full spatial conv before GlobalAveragePooling, giving the most
        detailed and accurate activation maps.
        Priority: Conv_1 → last Conv2D → first available layer
        """
        # Try MobileNetV2 specific layers first (best heatmaps)
        preferred = ["Conv_1", "Conv_1_bn", "out_relu",
                     "block_16_project", "block_16_project_BN"]
        for name in preferred:
            try:
                self.model.get_layer(name)
                return name
            except ValueError:
                continue

        # Fall back to last Conv2D
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        # Last resort — first layer
        return self.model.layers[0].name

    def generate(self, img_array):
        """
        Generate Grad-CAM heatmap.
        img_array: (1, 224, 224, 3) preprocessed float32
        Returns: (H, W) float32 heatmap 0–1
        """
        try:
            grad_model = tf.keras.Model(
                inputs  = self.model.inputs,
                outputs = [
                    self.model.get_layer(self.layer_name).output,
                    self.model.output,
                ]
            )

            with tf.GradientTape() as tape:
                inputs = tf.cast(img_array, tf.float32)
                conv_out, preds = grad_model(inputs)
                # Score for "cracked" class (sigmoid output)
                class_score = preds[:, 0]

            # Gradients of score w.r.t. conv feature maps
            grads = tape.gradient(class_score, conv_out)

            # Global average pool the gradients (weight each feature map)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight feature maps
            conv_out   = conv_out[0].numpy()
            pooled_grads = pooled_grads.numpy()

            # Weighted sum of feature maps
            for i in range(pooled_grads.shape[-1]):
                conv_out[:, :, i] *= pooled_grads[i]

            heatmap = np.mean(conv_out, axis=-1)

            # ReLU — only positive contributions matter
            heatmap = np.maximum(heatmap, 0)

            # Normalize to 0–1
            if heatmap.max() > 1e-8:
                heatmap = heatmap / heatmap.max()
            else:
                heatmap = np.zeros_like(heatmap)

            # Smooth slightly for cleaner visualization
            heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)

            return heatmap.astype(np.float32)

        except Exception as e:
            print(f"  ⚠️  Grad-CAM error: {e}")
            return np.zeros((7, 7), dtype=np.float32)
