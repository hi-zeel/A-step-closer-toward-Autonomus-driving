"""import pygame
from pygame import gfxdraw
import os
from PIL import Image
from PIL import ImageDraw

from gui_util import draw_dashed_line_delay

ROAD_HEIGHT = 250.0


COLOR = {"white": pygame.Color(255, 255, 255),
         "opaque_white": pygame.Color(255, 255, 255, 80),
         "text": pygame.Color(172, 199, 252),
         "dark_text": pygame.Color(57, 84, 137),
         "selection": [pygame.Color(172, 199, 252), pygame.Color(100, 149, 252)],
         "sky": pygame.Color(10, 10, 10),
         "gutter": pygame.Color(100, 100, 100),
         "red": pygame.Color(204, 0, 0),
         "bonus_a": pygame.Color(255, 78, 0),
         "bonus_b": pygame.Color(255, 178, 0),
         "green": pygame.Color(32, 76, 1),
         "black": pygame.Color(0, 0, 0),
         "tunnel": pygame.Color(38, 15, 8),
         "brown": pygame.Color(124, 69, 1)}


class AdvancedRoad:
    def __init__(self, surface, origin_x, origin_y, width, height, lane=6):
        self.surface = surface
        self.sky_x = 0
        self.sky_y = 0
        self.sky_height = 800 - ROAD_HEIGHT
        self.sky_width = width
        self.origin_x = origin_x # Road origin_x
        self.origin_y = origin_y # Road origin_y
        self.width = width # Road width
        self.height = height # Road height
        self.lane = lane

        self.sky_image = pygame.image.load(os.path.join('./advanced_view/images/sky_1010x550.png'))
        self.hill_image = pygame.image.load(os.path.join('./advanced_view/images/hill.png'))
        self.field_left_image = pygame.image.load(os.path.join('./advanced_view/images/field_left.png'))
        self.field_right_image = pygame.image.load(os.path.join('./advanced_view/images/field_right.png'))
        self.field_side_left_image = pygame.image.load(os.path.join('./advanced_view/images/field_side_left.png'))
        self.field_side_right_image = pygame.image.load(os.path.join('./advanced_view/images/field_side_right.png'))
        self.field_image = Image.open(os.path.join('./advanced_view/images/field.png')).convert("RGBA")
        # self.dirt_image = pygame.image.load(os.path.join('./advanced_view/images/dirt.png'))
        self.dirt_image = Image.open(os.path.join('./advanced_view/images/dirt.png')).convert("RGBA")
        self.subject_car_middle_image = pygame.image.load(os.path.join('./advanced_view/images/chev_rear.png'))
        self.subject_car_left_image = pygame.image.load(os.path.join('./advanced_view/images/chev_left.png'))
        self.subject_car_right_image = pygame.image.load(os.path.join('./advanced_view/images/chev_right.png'))
        self.object_car_middle_image = pygame.image.load(os.path.join('./advanced_view/images/civic_rear.png'))
        self.object_car_left_image = pygame.image.load(os.path.join('./advanced_view/images/civic_left.png'))
        self.object_car_right_image = pygame.image.load(os.path.join('./advanced_view/images/civic_right.png'))

        self.road_view = None

    def draw(self, frame, subject_car):
        lane = subject_car.lane
        while True:
            self.draw_sky(frame)
            self.draw_road_side(frame, self.lane)
            self.draw_road(frame, lane=self.lane)
            self.draw_cars(subject_car)
            self.draw_subject_car(self.lane - lane)
            if self.lane != lane:
                self.lane += 0.25 if lane > self.lane else - 0.25
            if abs(self.lane - lane) < 0.1:
                break
            pygame.event.poll()
            pygame.display.flip()
        self.lane = lane

    def draw_sky(self, frame):
        view = pygame.Surface((self.sky_width, self.sky_height))
        # Resize sky
        if frame % 40 == 0:
            self.sky_image = pygame.transform.scale(self.sky_image, (self.sky_image.get_size()[0]+2, self.sky_image.get_size()[1]+1))
            self.sky_image.blit(self.sky_image, ((-1, -2), (self.sky_width, self.sky_height)))
        view.blit(self.sky_image, ((0, 0), (self.sky_width, self.sky_height)))
        view.blit(self.hill_image, ((0, self.sky_height - 49), (self.sky_width, 49)))
        self.surface.blit(view, ((self.sky_x, self.sky_y), (self.sky_width, self.sky_height)))

    def draw_cars(self, subject_car):
        view = pygame.Surface((1010, ROAD_HEIGHT), pygame.SRCALPHA, 32)
        view = view.convert_alpha()
        camera_lane = subject_car.lane
        cars = subject_car.get_subjective_vision()
        for car in cars:
            lane = car[0]
            y = car[1]
            relative_y = y - 42
            if relative_y < 0:
                continue

            image = self.object_car_middle_image
            ratio = 231.0 / 328.0
            if lane != camera_lane:
                if lane > camera_lane:
                    image = self.object_car_right_image
                else:
                    image = self.object_car_left_image
            pt_top_left = (lane - 1) * 100.0 / 7 + 455
            pt_top_right = lane * 100.0 / 7 + 455
            pt_bottom_left = -337.0 * camera_lane + 673.33 + (lane - 1) * 337.0
            pt_bottom_right = -337.0 * camera_lane + 673.33 + lane * 337.0
            target_y = ROAD_HEIGHT * relative_y / 28.0
            target_bottom_left_x = pt_top_left + target_y / ROAD_HEIGHT * (pt_bottom_left - pt_top_left)
            target_bottom_right_x = pt_top_right + target_y / ROAD_HEIGHT * (pt_bottom_right - pt_top_right)
            image_width = int(target_bottom_right_x - target_bottom_left_x - 10)
            image_width = image_width if image_width > 0 else 0
            image_height = int(image_width * ratio)
            target_top_left_x = int(target_bottom_left_x + 5)
            target_top_left_y = int(target_y - image_height)
            a = pygame.transform.scale(image, (image_width, image_height))
            view.blit(a, (target_top_left_x, target_top_left_y))
        self.surface.blit(view, ((self.origin_x, self.origin_y), (self.width, self.height)))

    def draw_road_side(self, frame, lane):
        polygon_left = [
            (-337.0 * lane + 673.33 - 200, ROAD_HEIGHT),
            (455 - 20, 0),
            (0, 0),
            (0, ROAD_HEIGHT)]
        polygon_right = [
            (1010, ROAD_HEIGHT),
            (1010, 0),
            (555 + 20, 0),
            (-336.67 * lane + 3032 + 200, ROAD_HEIGHT)]
        maskIm = self.dirt_image.crop((0, 500 - (frame % 20) * 25, self.width, ROAD_HEIGHT + 500 - (frame % 20) * 25))
        pdraw = ImageDraw.Draw(maskIm)
        pdraw.polygon(polygon_left, fill=(255, 255, 255, 0), outline=(255, 255, 255, 0))
        pdraw.polygon(polygon_right, fill=(255, 255, 255, 0), outline=(255, 255, 255, 0))
        side = self.field_image.crop((0, 500 - (frame % 20) * 25, self.width, ROAD_HEIGHT + 500 - (frame % 20) * 25))
        side.paste(maskIm, (0, 0), mask=maskIm)
        side_pygame = pygame.image.fromstring(side.tobytes(), side.size, side.mode)
        self.surface.blit(side_pygame, ((self.origin_x, self.origin_y), (self.width, ROAD_HEIGHT)))

    def blit_mask(source, dest, destpos, mask, maskrect):
        """
        #Blit an source image to the dest surface, at destpos, with a mask, using
        #only the maskrect part of the mask.
        
"""
        tmp = source.copy()
        tmp.blit(mask, maskrect.topleft, maskrect, special_flags=pygame.BLEND_RGBA_MULT)
        dest.blit(tmp, destpos, dest.get_rect().clip(maskrect))

    def draw_road(self, frame, lane=1):
        self.road_view = pygame.Surface((1010, ROAD_HEIGHT), pygame.SRCALPHA, 32)
        self.road_view = self.road_view.convert_alpha()

        pygame.draw.polygon(self.road_view, COLOR['black'],
                            [
                                (-337.0 * lane + 673.33, ROAD_HEIGHT),
                                (455, 0),
                                (555, 0),
                                (-336.67 * lane + 3032, ROAD_HEIGHT)
                            ])

        left = -337.0 * lane + 673.33
        for i in range(8):
            if i == 0 or i == 7:
                gfxdraw.filled_polygon(self.road_view, (
                    (int(int(i * 100.0 / 7 + 455)), 0),
                    (int(int(i * 100.0 / 7 + 455)) + 7, 0),
                    (int(left) + 7, int(ROAD_HEIGHT)),
                    (int(left), int(ROAD_HEIGHT))
                ), COLOR['white'])
            else:
                draw_dashed_line_delay(self.road_view,
                                 COLOR['white'],
                                 (i * 100.0 / 7 + 455, 0),
                                 (left, ROAD_HEIGHT),
                                 width=5,
                                 dash_length=40,
                                 delay=frame % 3)
            left += 337.0

        self.surface.blit(self.road_view, ((self.origin_x, self.origin_y), (self.width, self.height)))

    def draw_subject_car(self, direction):
        image = self.subject_car_middle_image
        if direction > 0:
            image = self.subject_car_left_image
        elif direction < 0:
            image = self.subject_car_right_image
        self.surface.blit(image, (self.origin_x + 345, self.origin_y + 70))"""


# ============================================================
# advanced_view/road.py
# Lane-agnostic "advanced view" renderer with perspective.
# Works for any LANE_COUNT (e.g., 4), no hardcoded 7-lane math.
# Keeps your sky/field/car images, but computes all geometry
# from the panel rect and lane count.
# ============================================================

import os
import pygame
from pygame import gfxdraw
from PIL import Image, ImageDraw

from gui_util import draw_dashed_line_delay

# try to read lane-count from config; fallback = 7 (won't be used if ctor passes lane)
try:
    from config import LANE_COUNT as CFG_LANE_COUNT
except Exception:
    CFG_LANE_COUNT = 7

# Height (in px) of the road band inside this panel.
# The sky occupies (panel_height - ROAD_HEIGHT) above it.
ROAD_HEIGHT = 250

COLOR = {
    "white": pygame.Color(255, 255, 255),
    "opaque_white": pygame.Color(255, 255, 255, 80),
    "text": pygame.Color(172, 199, 252),
    "dark_text": pygame.Color(57, 84, 137),
    "selection": [pygame.Color(172, 199, 252), pygame.Color(100, 149, 252)],
    "sky": pygame.Color(10, 10, 10),
    "gutter": pygame.Color(100, 100, 100),
    "red": pygame.Color(204, 0, 0),
    "bonus_a": pygame.Color(255, 78, 0),
    "bonus_b": pygame.Color(255, 178, 0),
    "green": pygame.Color(32, 76, 1),
    "black": pygame.Color(0, 0, 0),
    "tunnel": pygame.Color(38, 15, 8),
    "brown": pygame.Color(124, 69, 1),
}


class AdvancedRoad:
    """
    Draws a sky + perspective road trapezoid + dashed lane separators,
    and projects car sprites along lane corridors based on their (lane,y)
    from subject_car.get_subjective_vision().

    Geometry is derived from:
      - panel rect (origin_x, origin_y, width, height)
      - lane_count (passed in or config)
    """

    def __init__(self, surface, origin_x, origin_y, width, height, lanes=None):
        # panel target
        self.surface = surface
        self.origin_x = int(origin_x)
        self.origin_y = int(origin_y)
        self.width = int(width)
        self.height = int(height)
        self.lanes = lanes

        if self.width <= 0 or self.height <= 0:
            raise ValueError("AdvancedRoad: invalid panel size.")

        # lane config
        self.lane_count = int(lanes) if lanes is not None else int(CFG_LANE_COUNT)
        if self.lane_count < 1:
            self.lane_count = 1

        # sky band height = remainder above road strip
        self.sky_height = max(0, self.height - ROAD_HEIGHT)
        self.sky_width = self.width

        # -------- load images (once) --------
        # store original images; we scale when drawing to fit the panel
        self.sky_image_orig = pygame.image.load(
            os.path.join("./advanced_view/images/sky_1010x550.png")
        )
        self.hill_image_orig = pygame.image.load(
            os.path.join("./advanced_view/images/hill.png")
        )
        self.field_left_image_orig = pygame.image.load(
            os.path.join("./advanced_view/images/field_left.png")
        )
        self.field_right_image_orig = pygame.image.load(
            os.path.join("./advanced_view/images/field_right.png")
        )
        self.field_side_left_image_orig = pygame.image.load(
            os.path.join("./advanced_view/images/field_side_left.png")
        )
        self.field_side_right_image_orig = pygame.image.load(
            os.path.join("./advanced_view/images/field_side_right.png")
        )

        # PIL textures used for simple scrolling side fields / dirt
        self.field_image = Image.open(
            os.path.join("./advanced_view/images/field.png")
        ).convert("RGBA")
        self.dirt_image = Image.open(
            os.path.join("./advanced_view/images/dirt.png")
        ).convert("RGBA")

        # car sprites
        self.subject_car_middle_image = pygame.image.load(
            os.path.join("./advanced_view/images/chev_rear.png")
        )
        self.subject_car_left_image = pygame.image.load(
            os.path.join("./advanced_view/images/chev_left.png")
        )
        self.subject_car_right_image = pygame.image.load(
            os.path.join("./advanced_view/images/chev_right.png")
        )
        self.object_car_middle_image = pygame.image.load(
            os.path.join("./advanced_view/images/civic_rear.png")
        )
        self.object_car_left_image = pygame.image.load(
            os.path.join("./advanced_view/images/civic_left.png")
        )
        self.object_car_right_image = pygame.image.load(
            os.path.join("./advanced_view/images/civic_right.png")
        )

        # a local surface for the road band (we draw into this then blit)
        self.road_view = None

        # trapezoid parameters (computed each draw)
        self._top_w_frac = 0.12   # top road width as fraction of panel width
        self._side_margin = 0.02  # bottom side margin fraction

    # ============================================================
    # public API
    # ============================================================
    def draw(self, frame, subject_car):
        """
        Render full panel (sky + road + cars) for this frame.
        """
        if self.surface is None:
            return

        self._draw_sky(frame)
        self._draw_road_sides(frame)      # scrolling side fields/dirt (simple)
        self._draw_road(frame)            # trapezoid asphalt + separators
        self._draw_cars(subject_car)      # other cars projected along lanes
        self._draw_subject_car(subject_car)  # subject car sprite (fixed near bottom)

    # ============================================================
    # sky + sides
    # ============================================================
    def _draw_sky(self, frame):
        """
        Draws the sky band. We scale sky + hills to the sky rect.
        """
        if self.sky_height <= 0:
            return

        sky_rect = pygame.Rect(self.origin_x, self.origin_y, self.sky_width, self.sky_height)

        # scale sky to band size (no cumulative growth)
        sky_img = pygame.transform.smoothscale(self.sky_image_orig, (self.sky_width, self.sky_height))
        self.surface.blit(sky_img, sky_rect)

        # hills stuck to bottom of sky band
        hill_h = min(49, self.sky_height)
        hill_img = pygame.transform.smoothscale(
            self.hill_image_orig,
            (self.sky_width, hill_h)
        )
        self.surface.blit(hill_img, (self.origin_x, self.origin_y + self.sky_height - hill_h))

    def _draw_road_sides(self, frame):
        """
        Very lightweight 'scrolling' side fields using the PIL textures.
        (This replaces the original complicated polygon masking.)
        """
        # build a PIL crop with simple vertical scroll
        y_scroll = (frame % 20) * 25
        # protect against textures smaller than needed: tile if necessary
        crop_h = ROAD_HEIGHT + 500
        crop = self.field_image.crop((0, 500 - y_scroll, self.width, 500 - y_scroll + crop_h))
        dirt = self.dirt_image.crop((0, 500 - y_scroll, self.width, 500 - y_scroll + crop_h))

        # alpha composite: dirt “cutouts” applied on field
        pdraw = ImageDraw.Draw(dirt)
        # we don't cut polygons anymore; keep the texture continuous
        # (Stable and lane-agnostic.)

        # paste dirt over field with its alpha
        field_with_dirt = crop.copy()
        field_with_dirt.paste(dirt, (0, 0), mask=dirt)

        # convert to pygame surface and blit behind the road
        side_pg = pygame.image.fromstring(field_with_dirt.tobytes(), field_with_dirt.size, field_with_dirt.mode)
        self.surface.blit(side_pg, (self.origin_x, self.origin_y + self.sky_height))

    # ============================================================
    # road (asphalt + lane separators)
    # ============================================================
    def _compute_trapezoid(self):
        """
        Returns trapezoid corners (in local road coordinates):
            TL, TR, BL, BR  (top-left, top-right, bottom-left, bottom-right)
        Local coords are relative to (0,0) at top-left of the road band.
        """
        w = self.width
        h = ROAD_HEIGHT

        top_w = max(40, int(w * self._top_w_frac))   # ensure at least some width
        bottom_margin = int(w * self._side_margin)
        bottom_left_x = bottom_margin
        bottom_right_x = w - bottom_margin

        top_left_x = (w - top_w) // 2
        top_right_x = top_left_x + top_w

        TL = (top_left_x, 0)
        TR = (top_right_x, 0)
        BL = (bottom_left_x, h)
        BR = (bottom_right_x, h)
        return TL, TR, BL, BR

    def _lane_boundary_x(self, i, TL, TR, BL, BR):
        """
        For lane-boundary index i in [0..lane_count], compute (top_x, bottom_x)
        at top and bottom edges. This subdivides the top/bottom widths evenly.
        """
        lane = self.lane_count
        # distances
        top_w = TR[0] - TL[0]
        bot_w = BR[0] - BL[0]
        frac = i / float(lane)
        top_x = TL[0] + frac * top_w
        bot_x = BL[0] + frac * bot_w
        return top_x, bot_x

    def _draw_road(self, frame):
        """
        Draw asphalt trapezoid and the lane boundaries
        (edges = solid, interior = dashed).
        """
        self.road_view = pygame.Surface((self.width, ROAD_HEIGHT), pygame.SRCALPHA, 32).convert_alpha()

        TL, TR, BL, BR = self._compute_trapezoid()

        # Asphalt polygon
        pygame.draw.polygon(self.road_view, COLOR["black"], [TL, TR, BR, BL])

        # Outer solid white edges
        pygame.draw.line(self.road_view, COLOR["white"], TL, BL, 6)
        pygame.draw.line(self.road_view, COLOR["white"], TR, BR, 6)

        # Interior dashed separators
        for i in range(1, self.lane_count):
            top_x, bot_x = self._lane_boundary_x(i, TL, TR, BL, BR)
            draw_dashed_line_delay(
                self.road_view,
                COLOR["white"],
                (top_x, 0),
                (bot_x, ROAD_HEIGHT),
                width=4,
                dash_length=40,
                delay=(frame % 3),
            )

        # Blit the road band at the bottom of the panel, under the sky
        self.surface.blit(self.road_view, (self.origin_x, self.origin_y + self.sky_height))

    # ============================================================
    # cars
    # ============================================================
    def _project_lane_span(self, lane_idx, t, TL, TR, BL, BR):
        """
        For lane index in [1..lane_count], and vertical parameter t in [0..1]
        (0=top, 1=bottom), compute the horizontal span (x_left, x_right)
        of that lane at that depth (interpolating between top and bottom).
        """
        # get boundary i-1 and i
        top_x_l, bot_x_l = self._lane_boundary_x(lane_idx - 1, TL, TR, BL, BR)
        top_x_r, bot_x_r = self._lane_boundary_x(lane_idx, TL, TR, BL, BR)
        x_left = top_x_l + t * (bot_x_l - top_x_l)
        x_right = top_x_r + t * (bot_x_r - top_x_r)
        return x_left, x_right

    def _draw_cars(self, subject_car):
        """
        Project other cars along lane corridors using get_subjective_vision().
        Each entry is (lane, y_bucket). We map y_bucket => depth 't', then
        compute the lane span and place/scales the sprite accordingly.
        """
        # camera lane (for left/right sprite selection)
        camera_lane = int(getattr(subject_car, "lane", 1))
        cars = subject_car.get_subjective_vision()

        # precompute trapezoid and helpers
        TL, TR, BL, BR = self._compute_trapezoid()

        # sprite aspect (rear car): width/height ratio
        # use your civic rear image as baseline ratio
        base_w, base_h = self.object_car_middle_image.get_size()
        car_ratio = base_h / float(base_w) if base_w else 1.5

        # map y_bucket (~ vision grid row) to panel depth:
        # original used: target_y = ROAD_HEIGHT * (y-42)/28
        def bucket_to_t(y_bucket):
            rel = (y_bucket - 42.0) / 28.0  # ~[-?, 1+]
            # clamp to [0,1] for drawing
            return max(0.0, min(1.0, rel))

        for lane, y_bucket in cars:
            if lane < 1 or lane > self.lane_count:
                continue

            t = bucket_to_t(y_bucket)
            if t <= 0.0:
                continue  # far ahead; too small to bother

            x_left, x_right = self._project_lane_span(lane, t, TL, TR, BL, BR)
            span = max(0, int(x_right - x_left - 10))
            if span <= 0:
                continue

            # choose sprite orientation relative to camera lane
            if lane > camera_lane:
                img = self.object_car_right_image
            elif lane < camera_lane:
                img = self.object_car_left_image
            else:
                img = self.object_car_middle_image

            # compute size: width proportional to lane span at depth;
            # height from aspect ratio
            img_w = span
            img_h = int(img_w * car_ratio)

            # y position: t along road height (then align car bottom)
            y = int(t * ROAD_HEIGHT) - img_h
            x = int(x_left + 5)

            if img_w <= 0 or img_h <= 0:
                continue

            scaled = pygame.transform.smoothscale(img, (img_w, img_h))
            self.surface.blit(scaled, (self.origin_x + x, self.origin_y + self.sky_height + y))

    def _draw_subject_car(self, subject_car):

        #Draw the subject (red) car at its actual lane position.
        #Uses left/right sprite if the car is switching lanes.
    

        # Default sprite = center
        img = self.subject_car_middle_image

        # Change sprite if last action is available
        if hasattr(subject_car, "last_action"):
            if subject_car.last_action == 'L':
                img = self.subject_car_left_image
            elif subject_car.last_action == 'R':
                img = self.subject_car_right_image

        # Car image size
        w, h = img.get_size()

        # --- Calculate horizontal position based on lane ---
        # Total number of lanes
        num_lanes = self.lanes

        # Width of one lane in pixels (projected)
        lane_width = self.width // num_lanes

        # Lane index of subject car (starts from 0 or 1 depending on your Car class)
        # If Car.lane starts from 1, subtract 1
        lane_index = subject_car.lane - 1

        # Compute X position so car aligns with lane
        x = self.origin_x + lane_index * lane_width + (lane_width - w) // 2

        # Y position (near bottom of road band)
        y = self.origin_y + self.sky_height + int(ROAD_HEIGHT * 0.28)

        # Draw car
        self.surface.blit(img, (x, y))
