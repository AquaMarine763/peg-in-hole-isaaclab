import math
import os


ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

PEG_RADIUS = 0.015
PEG_HEIGHT = 0.10
PEG_MASS = 0.1

HOLE_CLEARANCE = 0.0025
HOLE_INNER_RADIUS = PEG_RADIUS + HOLE_CLEARANCE
HOLE_DEPTH = 0.05
HOLE_BASE_WIDTH = 0.18
HOLE_BLOCK_HEIGHT = 0.14
HOLE_WALL_THICKNESS = 0.008
HOLE_FLOOR_THICKNESS = 0.01
HOLE_NUM_SEGMENTS = 32
HOLE_MASS = 1.0


def create_peg(radius: float, height: float, output_path: str):
    usda = f'''#usda 1.0
(
    defaultPrim = "Peg"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "Peg" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    bool physics:rigidBodyEnabled = true
    float physics:mass = {PEG_MASS}

    def Cylinder "Body" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        double radius = {radius}
        double height = {height}
        token axis = "Z"
        double3 xformOp:translate = (0, 0, {-height / 2.0})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        bool physics:collisionEnabled = true
        token physics:approximation = "convexHull"
        color3f[] primvars:displayColor = [(0.2, 0.8, 0.2)]
    }}
}}
'''
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(usda)


def slab(name: str, tx: float, ty: float, tz: float, sx: float, sy: float, sz: float, color: str):
    return f'''
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        double size = 1.0
        double3 xformOp:translate = ({tx}, {ty}, {tz})
        float3 xformOp:scale = ({sx}, {sy}, {sz})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        bool physics:collisionEnabled = true
        token physics:approximation = "convexHull"
        color3f[] primvars:displayColor = [{color}]
    }}'''


def create_hole(output_path: str):
    outer_radius = HOLE_INNER_RADIUS + HOLE_WALL_THICKNESS
    mid_radius = HOLE_INNER_RADIUS + HOLE_WALL_THICKNESS / 2.0
    seg_half_width = mid_radius * math.tan(math.pi / HOLE_NUM_SEGMENTS)
    side_width = HOLE_BASE_WIDTH / 2.0 - outer_radius

    slabs = [
        slab("LeftBlock", -(outer_radius + side_width / 2.0), 0.0, -HOLE_BLOCK_HEIGHT / 2.0, side_width, HOLE_BASE_WIDTH, HOLE_BLOCK_HEIGHT, "(0.6, 0.6, 0.65)"),
        slab("RightBlock", outer_radius + side_width / 2.0, 0.0, -HOLE_BLOCK_HEIGHT / 2.0, side_width, HOLE_BASE_WIDTH, HOLE_BLOCK_HEIGHT, "(0.6, 0.6, 0.65)"),
        slab("FrontBlock", 0.0, outer_radius + side_width / 2.0, -HOLE_BLOCK_HEIGHT / 2.0, outer_radius * 2.0, side_width, HOLE_BLOCK_HEIGHT, "(0.6, 0.6, 0.65)"),
        slab("BackBlock", 0.0, -(outer_radius + side_width / 2.0), -HOLE_BLOCK_HEIGHT / 2.0, outer_radius * 2.0, side_width, HOLE_BLOCK_HEIGHT, "(0.6, 0.6, 0.65)"),
        slab("HoleFloor", 0.0, 0.0, -(HOLE_DEPTH + HOLE_FLOOR_THICKNESS / 2.0), outer_radius * 2.0, outer_radius * 2.0, HOLE_FLOOR_THICKNESS, "(0.55, 0.55, 0.6)"),
    ]

    walls = []
    for i in range(HOLE_NUM_SEGMENTS):
        angle = 2.0 * math.pi * i / HOLE_NUM_SEGMENTS
        x = mid_radius * math.cos(angle)
        y = mid_radius * math.sin(angle)
        deg = math.degrees(angle)
        walls.append(f'''
    def Cube "Wall_{i}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        double size = 1.0
        double3 xformOp:translate = ({x}, {y}, {-HOLE_DEPTH / 2.0})
        float xformOp:rotateZ = {deg}
        float3 xformOp:scale = ({HOLE_WALL_THICKNESS}, {seg_half_width * 2}, {HOLE_DEPTH})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]
        bool physics:collisionEnabled = true
        token physics:approximation = "convexHull"
        color3f[] primvars:displayColor = [(0.65, 0.65, 0.7)]
    }}''')

    prim_block = "\n".join(slabs + walls)
    usda = f'''#usda 1.0
(
    defaultPrim = "Hole"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "Hole" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    bool physics:rigidBodyEnabled = true
    float physics:mass = {HOLE_MASS}
{prim_block}
}}
'''
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(usda)


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    create_peg(PEG_RADIUS, PEG_HEIGHT, os.path.join(ASSETS_DIR, "peg_cylinder.usda"))
    create_hole(os.path.join(ASSETS_DIR, "hole_block.usda"))
    print(f"Assets written to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
