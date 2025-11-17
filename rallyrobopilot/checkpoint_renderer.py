"""
checkpoint_renderer.py
Checkpoint renderer for RallyRobotPilot using Ursina Engine
Place this file in the rallyrobopilot/ directory (same level as Car, Track, etc.)
"""

import json
import math
from ursina import *


class CheckpointRenderer(Entity):
    """Renders checkpoints (with optional rings) and connecting lines for RallyRobotPilot."""

    def __init__(
        self,
        track_name="SimpleTrack",
        show_rings=False,
        zone_color=None,
        zone_alpha=0.25,
    ):
        """
        Args:
            track_name: name of the track (used to load JSON)
            show_rings: if False, hides ring segments & poles (zone-only view)
            zone_color: Color of the detection zone; default is red
            zone_alpha: Transparency of the detection zone
        """
        super().__init__()

        self.track_name = track_name
        self.show_rings = show_rings
        self.zone_color = zone_color or color.rgb(255, 0, 0)
        self.zone_alpha = zone_alpha

        self.checkpoints = []
        self.checkpoint_entities = []
        self.line_entities = []
        self.detection_zones = []  # references to the red squares only
        self.ring_parts = []  # references to ring segments & poles only

        self.current_checkpoint_idx = 0
        self.total_passes = 0
        self.lap_count = 0
        self.car = None

        # Small debounce & arming logic to avoid instant re-pass on respawn
        self._armed = True
        self._last_pass_t = 0.0
        self._pass_cooldown = 0.25  # seconds (used if you add perf_counter later)

        # Load checkpoints from JSON
        self._load_checkpoints()

        # Create visuals
        if self.checkpoints:
            self._create_checkpoint_visuals()
            print(f"✓ Loaded {len(self.checkpoints)} checkpoints for {track_name}")

    def _load_checkpoints(self):
        """Load checkpoints from JSON file"""
        checkpoint_file = f"assets/tracks/{self.track_name.lower()}_checkpoints.json"

        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
                self.checkpoints = data.get("checkpoints", [])
                print(
                    f"Checkpoint data loaded: {len(self.checkpoints)} checkpoints from {checkpoint_file}"
                )
        except FileNotFoundError:
            print(f"⚠ Checkpoint file not found: {checkpoint_file}")
            self.checkpoints = []
        except Exception as e:
            print(f"Error loading checkpoints: {e}")
            self.checkpoints = []

    def _create_checkpoint_visuals(self):
        """Create visual markers for each checkpoint"""
        for idx, cp in enumerate(self.checkpoints):
            pos = cp["pos"]
            radius = cp["radius"]

            group = Entity()

            # --- Detection zone (square) ---
            detection_zone = Entity(
                model="cube",
                scale=(radius * 2, 0.5, radius * 3),
                position=(pos[0], pos[1], pos[2]),
                color=self.zone_color,
                alpha=self.zone_alpha,
                parent=group,
            )
            detection_zone._tag = "zone"
            self.detection_zones.append(detection_zone)

            # --- Optional ring + poles (hidden when show_rings=False) ---
            if self.show_rings:
                segments = 32
                ring_thickness = 0.5

                # ring segments
                for i in range(segments):
                    angle1 = (i / segments) * 2 * math.pi
                    angle2 = ((i + 1) / segments) * 2 * math.pi

                    x1 = pos[0] + radius * math.cos(angle1)
                    z1 = pos[2] + radius * math.sin(angle1)
                    x2 = pos[0] + radius * math.cos(angle2)
                    z2 = pos[2] + radius * math.sin(angle2)

                    mid_x = (x1 + x2) / 2
                    mid_z = (z1 + z2) / 2

                    segment_length = math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)
                    segment_angle = math.degrees(math.atan2(z2 - z1, x2 - x1))

                    seg = Entity(
                        model="cube",
                        scale=(segment_length, ring_thickness, ring_thickness),
                        position=(mid_x, pos[1] + 2, mid_z),
                        rotation=(0, segment_angle, 0),
                        color=color.lime if idx == 0 else color.yellow,
                        parent=group,
                    )
                    seg._tag = "ring"
                    self.ring_parts.append(seg)

                # poles
                pole_height = 5
                for xoff in (-radius, radius):
                    pole = Entity(
                        model="cube",
                        scale=(0.3, pole_height, 0.3),
                        position=(pos[0] + xoff, pos[1] + pole_height / 2, pos[2]),
                        color=color.lime if idx == 0 else color.yellow,
                        parent=group,
                    )
                    pole._tag = "ring"
                    self.ring_parts.append(pole)

            self.checkpoint_entities.append(group)

    def set_car(self, car):
        """Set the car to track"""
        self.car = car

    def update(self):
        """Update checkpoint system - called every frame by Ursina"""
        if not self.checkpoints or not self.car:
            return

        car_pos = self.car.position
        self._check_checkpoint_pass(car_pos)

    # ---------- Reset & input ----------

    def reset(self):
        """Reset renderer state when the game/car resets."""
        self.current_checkpoint_idx = 0
        self.total_passes = 0
        self.lap_count = 0
        self._armed = False  # disarm until the car leaves the zone after respawn

        # Keep zones red & semi-transparent
        for dz in self.detection_zones:
            dz.color = self.zone_color
            dz.alpha = self.zone_alpha

        # Restore ring colors (only if they exist)
        if self.show_rings:
            for part in self.ring_parts:
                if hasattr(part, "color"):
                    part.color = color.yellow
            # first checkpoint parts lime, if present
            if self.checkpoint_entities:
                first_group = self.checkpoint_entities[0]
                for child in first_group.children:
                    if hasattr(child, "color") and getattr(child, "_tag", "") == "ring":
                        child.color = color.lime

        print("✓ CheckpointRenderer reset: idx=0, laps=0, passes=0")

    def input(self, key):
        # Use 'g up' to avoid conflicts with other handlers bound to 'g'
        if key == "g up":
            self.reset()

    # ---------- Pass detection ----------

    def _check_checkpoint_pass(self, car_pos: Vec3):
        """Check if car has passed through current checkpoint — box-based to match the red square."""
        idx = self.current_checkpoint_idx
        cp = self.checkpoints[idx]
        cp_pos = Vec3(cp["pos"][0], cp["pos"][1], cp["pos"][2])

        # fetch the red square entity we created for this checkpoint
        dz = self.detection_zones[idx]
        # half-extents from the visual scale (so if you change the square size, logic follows automatically)
        # Ursina exposes both vec3 scale and convenience attrs; support both:
        half_x = (dz.scale_x if hasattr(dz, "scale_x") else dz.scale[0]) * 0.5
        half_z = (dz.scale_z if hasattr(dz, "scale_z") else dz.scale[2]) * 0.5

        # 2D offset in XZ plane
        dx = car_pos.x - cp_pos.x
        dzed = car_pos.z - cp_pos.z

        inside = (abs(dx) <= half_x) and (abs(dzed) <= half_z)

        # re-arm logic so we don't double-trigger while staying inside
        margin = 0.5
        if not self._armed and (
            abs(dx) > half_x + margin or abs(dzed) > half_z + margin
        ):
            self._armed = True

        if self._armed and inside:
            self._on_checkpoint_passed()
            self._armed = False

    def _on_checkpoint_passed(self):
        """Handle checkpoint pass event"""
        cp = self.checkpoints[self.current_checkpoint_idx]
        self.total_passes += 1

        print(
            f"✓ Passed Checkpoint {self.current_checkpoint_idx + 1}/{len(self.checkpoints)} "
            f"at position: [{cp['pos'][0]:.1f}, {cp['pos'][1]:.1f}, {cp['pos'][2]:.1f}]"
        )

        # Lap completed?
        if self.current_checkpoint_idx == len(self.checkpoints) - 1:
            self.lap_count += 1
            print(f"🏁 LAP {self.lap_count} COMPLETED!")

        # Flash the checkpoint group (subtle)
        self._flash_checkpoint(self.current_checkpoint_idx)

        # Next checkpoint
        self.current_checkpoint_idx = (self.current_checkpoint_idx + 1) % len(
            self.checkpoints
        )

    def _flash_checkpoint(self, idx):
        """Create a flash effect on passed checkpoint"""
        if idx < len(self.checkpoint_entities):
            checkpoint = self.checkpoint_entities[idx]
            invoke(self._animate_flash, checkpoint, delay=0)

    def _animate_flash(self, checkpoint):
        """Animate checkpoint flash"""
        original_scale = checkpoint.scale
        checkpoint.animate_scale(original_scale * 1.2, duration=0.1)
        invoke(
            lambda: checkpoint.animate_scale(original_scale, duration=0.1), delay=0.1
        )

    # ---------- Info & cleanup ----------

    def get_current_checkpoint_info(self):
        """Get information about current checkpoint"""
        if not self.checkpoints:
            return None

        return {
            "current": self.current_checkpoint_idx + 1,
            "total": len(self.checkpoints),
            "lap": self.lap_count,
            "passes": self.total_passes,
            "position": self.checkpoints[self.current_checkpoint_idx]["pos"],
        }

    def cleanup(self):
        """Clean up checkpoint visuals"""
        for entity in self.checkpoint_entities + self.line_entities:
            destroy(entity)

        self.checkpoint_entities.clear()
        self.detection_zones.clear()
        self.ring_parts.clear()
        self.line_entities.clear()
        destroy(self)
