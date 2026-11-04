import os
import cv2
import numpy as np

def generate_samples(output_dirs):
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)

        # 1. Fresh Good Red Apple
        img_good_apple = np.full((400, 400, 3), (242, 242, 242), dtype=np.uint8)
        # Apple body (smooth gradient red)
        cv2.circle(img_good_apple, (200, 220), 120, (30, 30, 210), -1)
        cv2.circle(img_good_apple, (180, 210), 110, (40, 50, 230), -1)
        # Gloss highlight
        cv2.ellipse(img_good_apple, (160, 170), (45, 25), -30, 0, 360, (100, 110, 255), -1)
        # Stem & leaf
        cv2.line(img_good_apple, (200, 110), (195, 70), (20, 50, 80), 8)
        pts_leaf = np.array([[195, 80], [240, 60], [230, 95], [195, 80]])
        cv2.fillPoly(img_good_apple, [pts_leaf], (40, 160, 40))
        cv2.imwrite(os.path.join(out_dir, "sample_good_apple.jpg"), img_good_apple)

        # 2. Fresh Good Orange
        img_good_orange = np.full((400, 400, 3), (242, 242, 242), dtype=np.uint8)
        cv2.circle(img_good_orange, (200, 210), 115, (0, 140, 245), -1)
        # Green button stem
        cv2.circle(img_good_orange, (200, 105), 10, (30, 120, 30), -1)
        cv2.imwrite(os.path.join(out_dir, "sample_good_orange.jpg"), img_good_orange)

        # 3. Damaged Apple (Bruised / Cut)
        img_damaged = np.full((400, 400, 3), (242, 242, 242), dtype=np.uint8)
        cv2.circle(img_damaged, (200, 220), 120, (30, 30, 200), -1)
        # Bruise spot (dark brown discoloration)
        cv2.ellipse(img_damaged, (230, 200), (45, 30), 20, 0, 360, (20, 60, 100), -1)
        # Cut / fissure
        cv2.line(img_damaged, (140, 220), (180, 260), (10, 20, 60), 4)
        cv2.line(img_damaged, (145, 230), (160, 270), (15, 30, 75), 3)
        # Stem
        cv2.line(img_damaged, (200, 110), (195, 75), (20, 50, 80), 7)
        cv2.imwrite(os.path.join(out_dir, "sample_damaged_apple.jpg"), img_damaged)

        # 4. Rotten Apple (Fungal mold & dark necrotic decay)
        img_rotten = np.full((400, 400, 3), (242, 242, 242), dtype=np.uint8)
        cv2.circle(img_rotten, (200, 220), 120, (20, 40, 100), -1)
        # Sunken necrotic rot patch
        cv2.circle(img_rotten, (190, 210), 75, (10, 20, 45), -1)
        # Grayish mold fungal mycelium
        for _ in range(80):
            mx = np.random.randint(140, 240)
            my = np.random.randint(160, 260)
            if (mx-190)**2 + (my-210)**2 < 60**2:
                cv2.circle(img_rotten, (mx, my), np.random.randint(2, 6), (180, 190, 190), -1)
        cv2.imwrite(os.path.join(out_dir, "sample_rotten_apple.jpg"), img_rotten)

        # 5. Rotten Banana
        img_rotten_banana = np.full((400, 400, 3), (242, 242, 242), dtype=np.uint8)
        pts_banana = np.array([[80, 300], [120, 230], [200, 150], [300, 120], [330, 130], [240, 200], [150, 280], [90, 330]])
        cv2.fillPoly(img_rotten_banana, [pts_banana], (20, 35, 60))
        # Gray mold patches
        cv2.circle(img_rotten_banana, (180, 180), 30, (150, 160, 160), -1)
        cv2.circle(img_rotten_banana, (220, 160), 25, (140, 150, 150), -1)
        cv2.imwrite(os.path.join(out_dir, "sample_rotten_banana.jpg"), img_rotten_banana)

        # 6. Multi-fruit Platter / Conveyor Batch (4 distinct separated fruits)
        img_multi = np.full((450, 600, 3), (242, 242, 242), dtype=np.uint8)
        # Fruit 1: Good Apple (x=85, y=210)
        cv2.circle(img_multi, (85, 210), 55, (30, 30, 220), -1)
        cv2.circle(img_multi, (75, 190), 16, (60, 60, 245), -1)
        cv2.line(img_multi, (85, 155), (82, 135), (20, 50, 80), 4)

        # Fruit 2: Good Orange (x=215, y=210)
        cv2.circle(img_multi, (215, 210), 55, (0, 140, 245), -1)
        cv2.circle(img_multi, (215, 158), 6, (30, 120, 30), -1)

        # Fruit 3: Damaged Apple (x=350, y=210)
        cv2.circle(img_multi, (350, 210), 55, (30, 30, 200), -1)
        cv2.ellipse(img_multi, (365, 200), (22, 14), 15, 0, 360, (20, 60, 100), -1)
        cv2.line(img_multi, (330, 210), (345, 230), (10, 20, 60), 3)

        # Fruit 4: Rotten Fruit (x=485, y=210)
        cv2.circle(img_multi, (485, 210), 55, (20, 35, 70), -1)
        cv2.circle(img_multi, (480, 205), 32, (10, 20, 45), -1)
        for _ in range(25):
            mx = np.random.randint(460, 505)
            my = np.random.randint(190, 230)
            cv2.circle(img_multi, (mx, my), 2, (170, 180, 180), -1)

        cv2.imwrite(os.path.join(out_dir, "sample_multi_fruits.jpg"), img_multi)

    print("Sample fruit images generated successfully!")

if __name__ == "__main__":
    import sys
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = [
        os.path.join(base, "sample_images"),
        os.path.join(base, "backend", "static", "samples")
    ]
    generate_samples(dirs)
