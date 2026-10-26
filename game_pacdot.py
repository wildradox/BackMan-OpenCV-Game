import os
import random
import sys

import cv2
import numpy as np
import pygame

import hand_control_opencv as hc

# ================= Konfigurasi =================
CELL = 42                      # ukuran 1 petak labirin (px) -- naikkan/turunkan buat window lebih besar/kecil
FPS = 30
MOVE_EVERY_MS = 170            # kecepatan auto-move karakter (makin kecil makin cepat)
ENEMY_MOVE_EVERY_MS = 260      # musuh gerak lebih lambat dari player (biar fair)
LIVES_START = 3
CAM_PREVIEW_W = 380            # lebar preview kamera kecil di pojok game (diperbesar biar kelihatan jelas)

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

COLOR_BG = (15, 15, 25)
COLOR_WALL = (40, 70, 160)
COLOR_PATH = (10, 10, 18)
COLOR_DOT = (255, 210, 60)
COLOR_PLAYER = (255, 235, 70)
COLOR_ENEMY = (235, 60, 60)
COLOR_TEXT = (240, 240, 240)

# '#'=dinding, '.'=jalan+dot, ' '=jalan kosong, 'P'=start player, 'E'=start enemy
MAZE_LAYOUT = [
    "###############",
    "#P............#",
    "#.###.#####.#.#",
    "#...#.#...#.#.#",
    "###.#.#.###.#.#",
    "#...#.....#...#",
    "#.#########.###",
    "#...........#.#",
    "#.#########.#.#",
    "#...........E.#",
    "###############",
]

DIR_VECTORS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    None: (0, 0),
}


# ================= Aset (opsional) =================
def load_image(filename, size):
    """Coba load gambar dari folder assets/. Kalau tidak ada -> None (fallback bentuk warna)."""
    path = os.path.join(ASSET_DIR, filename)
    if os.path.isfile(path):
        try:
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.smoothscale(img, size)
        except Exception as e:
            print(f"[WARNING] Gagal memuat assets/{filename}: {e}")
    return None


def cv2_frame_to_surface(frame_bgr):
    """Konversi frame OpenCV (BGR) jadi pygame.Surface untuk ditampilkan di jendela game."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))


# ================= Labirin =================
class Maze:
    def __init__(self, layout):
        self.rows = len(layout)
        self.cols = len(layout[0])
        self.grid = [list(row) for row in layout]
        self.player_start = (1, 1)
        self.enemy_start = (1, 1)

        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "P":
                    self.player_start = (c, r)
                    self.grid[r][c] = "."
                elif ch == "E":
                    self.enemy_start = (c, r)
                    self.grid[r][c] = "."

        self.dots = self._reachable_dots()

    def is_wall(self, col, row):
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return True
        return self.grid[row][col] == "#"

    def _reachable_dots(self):
        """
        BFS dari posisi start player -> hanya sel yang BENAR-BENAR bisa
        dicapai yang dijadikan dot. Ini menjamin game selalu bisa dimenangkan
        (dot tidak pernah ditaruh di kantong labirin yang tidak terjangkau).
        """
        from collections import deque as _dq

        visited = set()
        q = _dq([self.player_start])
        visited.add(self.player_start)

        while q:
            cx, cy = q.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not self.is_wall(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        visited.discard(self.player_start)
        return visited

    def pixel_size(self):
        return self.cols * CELL, self.rows * CELL


# ================= Entitas =================
class Entity:
    def __init__(self, col, row):
        self.col, self.row = col, row
        self.direction = None
        self.last_move_time = 0

    def pos_px(self):
        return self.col * CELL, self.row * CELL

    def try_move(self, maze: Maze, direction, now_ms, interval_ms):
        if direction is None:
            return False
        if now_ms - self.last_move_time < interval_ms:
            return False
        dx, dy = DIR_VECTORS[direction]
        nx, ny = self.col + dx, self.row + dy
        if not maze.is_wall(nx, ny):
            self.col, self.row = nx, ny
            self.direction = direction
            self.last_move_time = now_ms
            return True
        return False


class Player(Entity):
    def __init__(self, col, row):
        super().__init__(col, row)
        self.next_direction = None

    def update(self, maze: Maze, hand_command, now_ms):
        # Perintah tangan (kalau ada) jadi niat arah berikutnya
        if hand_command is not None:
            self.next_direction = hand_command

        # Coba belok ke next_direction dulu (kalau jalannya terbuka)
        if self.next_direction is not None:
            if self.try_move(maze, self.next_direction, now_ms, MOVE_EVERY_MS):
                return
        # Kalau tidak bisa belok, lanjut jalan ke arah sekarang (auto-move)
        self.try_move(maze, self.direction, now_ms, MOVE_EVERY_MS)


class Enemy(Entity):
    def update(self, maze: Maze, now_ms):
        if self.direction is None:
            self.direction = random.choice(list(DIR_VECTORS.keys())[:-1])

        opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        options = [d for d in ("UP", "DOWN", "LEFT", "RIGHT")
                   if not maze.is_wall(self.col + DIR_VECTORS[d][0], self.row + DIR_VECTORS[d][1])]
        if not options:
            return

        # Lebih suka lanjut arah sekarang atau belok, hindari balik arah kalau masih ada opsi lain
        non_reverse = [d for d in options if d != opposite.get(self.direction)]
        pick_from = non_reverse if non_reverse else options

        if self.direction not in pick_from or random.random() < 0.25:
            self.direction = random.choice(pick_from)

        self.try_move(maze, self.direction, now_ms, ENEMY_MOVE_EVERY_MS)


# ================= Game =================
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Pac-Dot Collector - Kontrol Tangan (OpenCV)")

        self.maze = Maze(MAZE_LAYOUT)
        maze_w, maze_h = self.maze.pixel_size()
        self.hud_h = 90
        self.cam_h = int(CAM_PREVIEW_W * hc.CAM_HEIGHT / hc.CAM_WIDTH)

        self.screen_w = max(maze_w, CAM_PREVIEW_W + 20)
        self.screen_h = maze_h + self.hud_h + self.cam_h + 20
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.font_big = pygame.font.SysFont("consolas", 34, bold=True)

        self.player_img = load_image("player.png", (CELL, CELL))
        self.enemy_img = load_image("enemy.png", (CELL, CELL))
        self.dot_img = load_image("dot.png", (CELL // 2, CELL // 2))
        self.wall_img = load_image("wall.png", (CELL, CELL))
        self.bg_img = load_image("bg.png", (maze_w, maze_h))

        self.cap = hc.open_camera(0)
        self.reset()

    def reset(self):
        self.maze = Maze(MAZE_LAYOUT)
        px, py = self.maze.player_start
        ex, ey = self.maze.enemy_start
        self.player = Player(px, py)
        self.enemy = Enemy(ex, ey)
        self.score = 0
        self.lives = LIVES_START
        self.state = "WAIT_CALIBRATION" if not hc.calibrated else "PLAYING"
        self.message = None

    # -------- Render labirin --------
    def draw_maze(self, surf):
        ox, oy = 0, self.hud_h
        if self.bg_img:
            surf.blit(self.bg_img, (ox, oy))
        else:
            surf.fill(COLOR_PATH, (ox, oy, *self.maze.pixel_size()))

        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                x, y = ox + c * CELL, oy + r * CELL
                if self.maze.is_wall(c, r):
                    if self.wall_img:
                        surf.blit(self.wall_img, (x, y))
                    else:
                        pygame.draw.rect(surf, COLOR_WALL, (x, y, CELL, CELL))
                        pygame.draw.rect(surf, (20, 35, 90), (x, y, CELL, CELL), 1)

        for (c, r) in self.maze.dots:
            x, y = ox + c * CELL + CELL // 2, oy + r * CELL + CELL // 2
            if self.dot_img:
                rect = self.dot_img.get_rect(center=(x, y))
                surf.blit(self.dot_img, rect)
            else:
                pygame.draw.circle(surf, COLOR_DOT, (x, y), max(CELL // 8, 3))

        # Enemy
        ex, ey = self.enemy.pos_px()
        ex, ey = ox + ex, oy + ey
        if self.enemy_img:
            surf.blit(self.enemy_img, (ex, ey))
        else:
            pygame.draw.circle(surf, COLOR_ENEMY, (ex + CELL // 2, ey + CELL // 2), CELL // 2 - 2)

        # Player
        px_, py_ = self.player.pos_px()
        px_, py_ = ox + px_, oy + py_
        if self.player_img:
            surf.blit(self.player_img, (px_, py_))
        else:
            pygame.draw.circle(surf, COLOR_PLAYER, (px_ + CELL // 2, py_ + CELL // 2), CELL // 2 - 2)

    def draw_hud(self, surf, hand_result):
        surf.fill((25, 25, 40), (0, 0, self.screen_w, self.hud_h))
        score_txt = self.font.render(f"Skor: {self.score}  |  Nyawa: {self.lives}", True, COLOR_TEXT)
        surf.blit(score_txt, (14, 10))

        cmd = hand_result["command"] if hand_result else None
        cmd_txt = self.font.render(f"Perintah tangan: {cmd}", True, (120, 230, 120) if cmd else (160, 160, 160))
        surf.blit(cmd_txt, (14, 38))

        orient = hand_result["orientation"] if hand_result else None
        orient_calib_done = hc._palm_features is not None and hc._back_features is not None
        orient_color = (120, 230, 120) if orient_calib_done else (235, 160, 60)
        orient_txt = self.font.render(
            f"Orientasi: {orient}  [kalibrasi: {'OK' if orient_calib_done else 'tekan P & B'}]",
            True, orient_color)
        surf.blit(orient_txt, (14, 64))

        if self.state == "WAIT_CALIBRATION":
            info = self.font.render("Taruh telapak di ZONA TANGAN (preview kamera) lalu tekan 'c'", True, (255, 220, 80))
            surf.blit(info, (320, 10))

    def draw_camera_preview(self, surf, frame_display):
        preview = cv2.resize(frame_display, (CAM_PREVIEW_W, self.cam_h))
        preview_surf = cv2_frame_to_surface(preview)
        px = self.screen_w - CAM_PREVIEW_W - 10
        py = self.screen_h - self.cam_h - 10
        surf.blit(preview_surf, (px, py))
        pygame.draw.rect(surf, (90, 90, 110), (px - 2, py - 2, CAM_PREVIEW_W + 4, self.cam_h + 4), 2)

    def draw_center_message(self, surf, text, color):
        txt = self.font_big.render(text, True, color)
        rect = txt.get_rect(center=(self.screen_w // 2, self.hud_h + (self.screen_h - self.hud_h - self.cam_h) // 2))
        pad = 16
        pygame.draw.rect(surf, (0, 0, 0), rect.inflate(pad * 2, pad * 2))
        pygame.draw.rect(surf, color, rect.inflate(pad * 2, pad * 2), 2)
        surf.blit(txt, rect)

    # -------- Loop utama --------
    def run(self):
        print("[INFO] Tekan 'c' untuk kalibrasi warna kulit, panah untuk kontrol manual, 'r' restart, ESC keluar.")
        running = True
        while running:
            now_ms = pygame.time.get_ticks()

            ok, frame = self.cap.read()
            hand_result = None
            if ok:
                frame = cv2.flip(frame, 1)
                frame_display = frame.copy()
                hand_result = hc.detect_hand(frame)
                # show_text=False: preview-nya bakal diperkecil, teks detail
                # biarin tampil di HUD resolusi penuh di atas saja supaya
                # tetap terbaca (bukan ikut mengecil dan jadi tidak jelas).
                hc.draw_debug_overlay(frame_display, hand_result, show_text=False)
            else:
                frame_display = np.zeros((hc.CAM_HEIGHT, hc.CAM_WIDTH, 3), dtype=np.uint8)

            keys_direction = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c and ok:
                        hc.calibrate_skin_color(frame)
                        if self.state == "WAIT_CALIBRATION":
                            self.state = "PLAYING"
                    elif event.key == pygame.K_p and hand_result and hand_result["orientation_features"] is not None:
                        hc.calibrate_orientation(hand_result["orientation_features"], "palm")
                    elif event.key == pygame.K_b and hand_result and hand_result["orientation_features"] is not None:
                        hc.calibrate_orientation(hand_result["orientation_features"], "back")
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_UP:
                        keys_direction = "UP"
                    elif event.key == pygame.K_DOWN:
                        keys_direction = "DOWN"
                    elif event.key == pygame.K_LEFT:
                        keys_direction = "LEFT"
                    elif event.key == pygame.K_RIGHT:
                        keys_direction = "RIGHT"

            if self.state == "WAIT_CALIBRATION" and hc.calibrated:
                self.state = "PLAYING"

            command = keys_direction or (hand_result["command"] if hand_result else None)

            if self.state == "PLAYING":
                self.player.update(self.maze, command, now_ms)
                self.enemy.update(self.maze, now_ms)

                p_cell = (self.player.col, self.player.row)
                if p_cell in self.maze.dots:
                    self.maze.dots.discard(p_cell)
                    self.score += 10

                if (self.player.col, self.player.row) == (self.enemy.col, self.enemy.row):
                    self.lives -= 1
                    if self.lives <= 0:
                        self.state = "GAME_OVER"
                    else:
                        self.player.col, self.player.row = self.maze.player_start
                        self.player.direction = None
                        self.player.next_direction = None

                if not self.maze.dots:
                    self.state = "WIN"

            # ---- Gambar ----
            self.screen.fill(COLOR_BG)
            self.draw_hud(self.screen, hand_result)
            self.draw_maze(self.screen)
            self.draw_camera_preview(self.screen, frame_display)

            if self.state == "WAIT_CALIBRATION":
                self.draw_center_message(self.screen, "KALIBRASI DULU (tekan C)", (255, 210, 60))
            elif self.state == "GAME_OVER":
                self.draw_center_message(self.screen, "GAME OVER - tekan R", (235, 60, 60))
            elif self.state == "WIN":
                self.draw_center_message(self.screen, "KAMU MENANG! - tekan R", (80, 230, 120))

            pygame.display.flip()
            self.clock.tick(FPS)

        self.cap.release()
        pygame.quit()


def main():
    try:
        Game().run()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
