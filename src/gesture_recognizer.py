
"""GestureFlow AI - Improved rule-based gesture recognition."""

import math


class GestureRecognizer:
    FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

    # Tune these only after testing your webcam.
    PINCH_PALM_RATIO = 0.22
    THUMB_UP_MARGIN = 0.02

    @staticmethod
    def _landmarks(hand):
        return list(hand.landmark) if hasattr(hand, "landmark") else list(hand)

    @staticmethod
    def _distance(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _finger_extended(points, tip, pip):
        return points[tip].y < points[pip].y

    def recognize(self, hand):
        p = self._landmarks(hand)

        if len(p) != 21:
            return {
                "gesture": "Unknown",
                "pinch_distance": None,
                "fingers": {},
            }

        # Finger states: index, middle, ring, pinky.
        fingers = {
            name: self._finger_extended(
                p, tip, self.FINGER_PIPS[name]
            )
            for name, tip in self.FINGER_TIPS.items()
        }

        # Palm scale makes pinch less dependent on camera distance.
        palm_size = self._distance(p[0], p[9])
        if palm_size < 1e-6:
            palm_size = 1e-6

        pinch_distance = self._distance(p[4], p[8])
        pinch_ratio = pinch_distance / palm_size

        # A thumb pointing upward should have its tip above its
        # IP joint and MCP joint in the image.
        thumb_up = (
            p[4].y < p[3].y - self.THUMB_UP_MARGIN
            and p[3].y < p[2].y + 0.03
        )

        other_fingers_curled = not any(fingers.values())

        # 1. Prioritize thumbs-up so it is not mistaken for a pinch.
        if thumb_up and other_fingers_curled:
            gesture = "Thumbs Up"

        # 2. Recognize a pinch only when fingertips are very close.
        elif pinch_ratio < self.PINCH_PALM_RATIO:
            gesture = "Pinch"

        # 3. Fist: no fingers extended and thumb not pointing up.
        elif other_fingers_curled:
            gesture = "Fist"

        # 4. Open palm.
        elif all(fingers.values()):
            gesture = "Open Palm"

        # 5. Peace sign.
        elif (
            fingers["index"]
            and fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
        ):
            gesture = "Peace Sign"

        # 6. Pointing.
        elif (
            fingers["index"]
            and not fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
        ):
            gesture = "Pointing"

        else:
            gesture = "Unknown"

        return {
            "gesture": gesture,
            "pinch_distance": pinch_distance,
            "pinch_ratio": pinch_ratio,
            "fingers": fingers,
        }
