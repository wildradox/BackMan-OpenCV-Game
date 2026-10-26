# BackMan - Hand-Controlled Mini Game
A maze mini-game (inspired by the classic Pac-Man concept) interactively controlled using **hand color tracking** via webcam and **OpenCV**, built with **Python** and the **Pygame** framework.

---

##  Game Description
**Pac-Dot Collector** is a casual arcade game where the player acts as a character tasked with collecting all gold dots within a maze. The challenge is to dodge enemies patrolling randomly across the map.
What makes this game unique is its control scheme: **no keyboard or mouse required**, instead controlled using hand gestures in front of a web camera in real-time! (With optional arrow key support as a backup).

---

## Controls & Interaction
This game uses the `hand_control_opencv` helper module to track the player's hand position and directional movement.

1. **Skin/Hand Color Calibration:**
   * Position your hand inside the camera preview zone in the corner of the screen.
   * Press the **`C`** key on your keyboard while your hand is in front of the camera to sample and calibrate your hand color.
2. **Hand Movement Control:**
   * Move your hand according to the directional detection configuration in the `hand_control_opencv` module.
3. **Alternative Controls (Keyboard):**
   * If you prefer conventional play, you can use the **Arrow Keys (`UP`, `DOWN`, `LEFT`, `RIGHT`)** on your keyboard.
4. **Other Shortcuts:**
   * **`R`** : Restart the game (upon Game Over or Win).
   * **`ESC`** : Quit the game.

---

## How to Run the Game

### Prerequisites
Make sure you have **Python 3.x** installed on your system. You will need the following Python packages:
* `opencv-python` (for webcam capture and image processing)
* `numpy` (for matrix and mathematical operations)
* `pygame` (for rendering the game graphics and handling game loops)

### Installation Steps
1. Clone or download your repository containing your game script and the helper module (`hand_control_opencv.py`).
2. Open your terminal or command prompt inside the project directory.
3. Install the required dependencies using pip:
   ```bash
   pip install opencv-python numpy pygame
   ```
4. Ensure that the helper module (`hand_control_opencv.py`) and your main game script are in the same folder.
5. Run the game via terminal:
   ```bash
   python your_game_filename.py
   ```
---

## Theory: How Hand Gesture Tracking Works

The underlying mechanics bridge the gap between low-level Computer Vision processing and high-level Game State Management. Here is the technical breakdown of how the hand tracking system operates:

1. Color Space Conversion (BGR to HSV)
* **The Theory:** Standard webcams capture frames in the BGR (Blue, Green, Red) color space, which is heavily susceptible to lighting changes, shadows, and highlights. To resolve this, frames are converted into HSV (Hue, Saturation, Value). HSV separates color chromaticity (Hue) from lighting intensity (Value), making object/skin tracking significantly more robust against ambient lighting variations.
* **Implementation:** During calibration (`hc.calibrate_skin_color()`), the application samples the specific pixel values of the user's hand, establishing lower and upper HSV boundaries.

2. Thresholding and Binary Masking (`cv2.inRange`)
- The Theory: Once the target color boundaries are defined, OpenCV applies a filter across the incoming frame using `cv2.inRange()`.
- Implementation: This operation creates a binary mask a black-and-white image where pixels matching the hand color turn white (255), and background pixels turn black (0). This effectively performs background subtraction, isolating the hand from its environment.

3. Contour Detection & Image Moments
- The Theory: To locate the isolated hand object mathematically, OpenCV uses topological structural analysis via `cv2.findContours()` to outline the white regions on the binary mask.
- Implementation: The center of mass of the hand blob is calculated using Image Moments (M). Specifically, spatial moments M_10 and M_01 divided by the area moment M_00 yield the exact centroid coordinates (x, y) of the hand:
  - Cx = M_10 / M_00
  - Cy = M_01 / M_00
  Tracking how these coordinates shift across successive frames allows the system to determine directional vectors (Up, Down, Left, Right).

### 4. Game Logic & Pathfinding Architecture
* **Breadth-First Search (BFS) for Dot Generation:**
  * To guarantee that the game is always winnable, the `Maze` class implements a BFS (Breadth-First Search) algorithm starting from the player's initial spawn point (`_reachable_dots()`). Only cells that are genuinely reachable through open pathways are populated with collectable dots, preventing items from spawning inside unreachable dead ends.
* **Time-Delta Game Loop (`now_ms`):**
  * Player and enemy movements do not rely directly on raw frame rates (FPS), which can fluctuate based on hardware performance. Instead, they run on a delta-time/millisecond accumulator (`MOVE_EVERY_MS` and `ENEMY_MOVE_EVERY_MS`), ensuring smooth and consistent gameplay speeds across different machines.
