import cv2
import numpy as np

# Read input files
input_files = ["image1.jpeg", "image2.jpeg", "image3.jpeg"]

# Padding settings
y_padding = 100
x_padding = 100

# Read anchor coordinates
anchor_pairs = []
with open("coordinates.csv", "r", encoding="utf-8") as f:
    for line in f:
        x1, y1, x2, y2 = map(float, line.strip().split(","))
        anchor_pairs.append(np.array([[x1, y1], [x2, y2]]))

pair1, pair2, pair3, pair4, pair5, pair6 = anchor_pairs[:6]

# Print coordinate pairs
print("Img1:")
print(f"\tpair1: ({pair1[0,0]}, {pair1[0,1]}) <-> ({pair1[1,0]}, {pair1[1,1]})")
print(f"\tpair2: ({pair2[0,0]}, {pair2[0,1]}) <-> ({pair2[1,0]}, {pair2[1,1]})")
print(f"\tpair3: ({pair3[0,0]}, {pair3[0,1]}) <-> ({pair3[1,0]}, {pair3[1,1]})")
print("Img2:")
print(f"\tpair4: ({pair4[0,0]}, {pair4[0,1]}) <-> ({pair4[1,0]}, {pair4[1,1]})")
print(f"\tpair5: ({pair5[0,0]}, {pair5[0,1]}) <-> ({pair5[1,0]}, {pair5[1,1]})")
print(f"\tpair6: ({pair6[0,0]}, {pair6[0,1]}) <-> ({pair6[1,0]}, {pair6[1,1]})")

# Create matrices for transformation
points1 = np.float32(
    [
        [pair1[0, 0], pair1[0, 1], 1],
        [pair2[0, 0], pair2[0, 1], 1],
        [pair3[0, 0], pair3[0, 1], 1],
    ]
)

points2 = np.float32(
    [
        [pair4[0, 0], pair4[0, 1], 1],
        [pair5[0, 0], pair5[0, 1], 1],
        [pair6[0, 0], pair6[0, 1], 1],
    ]
)

x_vals1 = np.float32([pair1[1, 0], pair2[1, 0], pair3[1, 0]])
y_vals1 = np.float32([pair1[1, 1], pair2[1, 1], pair3[1, 1]])
x_vals2 = np.float32([pair4[1, 0], pair5[1, 0], pair6[1, 0]])
y_vals2 = np.float32([pair4[1, 1], pair5[1, 1], pair6[1, 1]])

# Solve using OpenCV
ret, x_dst1 = cv2.solve(points1, x_vals1, flags=cv2.DECOMP_LU)
ret, y_dst1 = cv2.solve(points1, y_vals1, flags=cv2.DECOMP_LU)
ret, x_dst2 = cv2.solve(points2, x_vals2, flags=cv2.DECOMP_LU)
ret, y_dst2 = cv2.solve(points2, y_vals2, flags=cv2.DECOMP_LU)

print(f"Img1:")
print(f"\ta: {x_dst1[0,0]:.6f}, b: {x_dst1[1,0]:.6f}, c: {x_dst1[2,0]:.6f}")
print(f"\td: {y_dst1[0,0]:.6f}, e: {y_dst1[1,0]:.6f}, f: {y_dst1[2,0]:.6f}")
print(f"Img2:")
print(f"\ta: {x_dst2[0,0]:.6f}, b: {x_dst2[1,0]:.6f}, c: {x_dst2[2,0]:.6f}")
print(f"\td: {y_dst2[0,0]:.6f}, e: {y_dst2[1,0]:.6f}, f: {y_dst2[2,0]:.6f}")


def interpolate(img, origin_x, origin_y):
    """Bilinear interpolation for image"""
    x_1 = int(np.floor(origin_x))
    x_2 = x_1 + 1
    y_1 = int(np.floor(origin_y))
    y_2 = y_1 + 1

    if x_1 < 0 or x_2 >= img.shape[1] or y_1 < 0 or y_2 >= img.shape[0]:
        return np.zeros(3, dtype=np.uint8)

    dx = origin_x - x_1
    dy = origin_y - y_1

    c1 = img[y_1, x_1]
    c2 = img[y_1, x_2]
    c3 = img[y_2, x_1]
    c4 = img[y_2, x_2]

    return (
        c1 * (1 - dx) * (1 - dy)
        + c2 * dx * (1 - dy)
        + c3 * (1 - dx) * dy
        + c4 * dx * dy
    ).astype(np.uint8)


def find_boundary(img):
    """找出邊界，回傳最大 x, y 和最小 y"""
    non_zero = np.nonzero(np.any(img != 0, axis=2))
    min_boundary_y = np.min(non_zero[0]) if len(non_zero[0]) > 0 else 0
    max_boundary_y = np.max(non_zero[0]) if len(non_zero[0]) > 0 else img.shape[0]
    max_boundary_x = np.max(non_zero[1]) if len(non_zero[1]) > 0 else img.shape[1]
    return max_boundary_x, max_boundary_y, min_boundary_y


# Read images
img1 = cv2.imread(input_files[0])
img2 = cv2.imread(input_files[1])
img3 = cv2.imread(input_files[2])

# Create intermediate image
intermediate = np.zeros(
    (img2.shape[0] + 2 * y_padding, int(img2.shape[1] + img3.shape[1] * 1.5), 3),
    dtype=np.uint8,
)

# Transform img3 to img2 space
for new_y in range(-y_padding, img2.shape[0] + 2 * y_padding):
    for new_x in range(img2.shape[1] + img3.shape[1] + x_padding):
        x = x_dst2[0, 0] * new_x + x_dst2[1, 0] * new_y + x_dst2[2, 0]
        y = y_dst2[0, 0] * new_x + y_dst2[1, 0] * new_y + y_dst2[2, 0]

        if 0 <= x < img3.shape[1] and 0 <= y < img3.shape[0]:
            intermediate[new_y + y_padding, new_x] = interpolate(img3, x, y)

# Copy img2 to intermediate
intermediate[y_padding : y_padding + img2.shape[0], 0 : img2.shape[1]] = img2

# Find boundary of intermediate result
max_x, max_y, min_y = find_boundary(intermediate)
intermediate = intermediate[:max_y, :max_x]

# Create final output
output_h = max(img1.shape[0], intermediate.shape[0]) + 2 * y_padding
output_w = img1.shape[1] + intermediate.shape[1] + x_padding
final_output = np.zeros((output_h, output_w, 3), dtype=np.uint8)

# Transform intermediate to img1 space
for new_y in range(-y_padding, output_h):
    for new_x in range(output_w):
        x = x_dst1[0, 0] * new_x + x_dst1[1, 0] * new_y + x_dst1[2, 0]
        y = y_dst1[0, 0] * new_x + y_dst1[1, 0] * new_y + y_dst1[2, 0]

        if 0 <= y < intermediate.shape[0] and 0 <= x < intermediate.shape[1]:
            final_output[new_y, new_x] = interpolate(intermediate, x, y)

# Copy img1 to final output
final_output[y_padding : y_padding + img1.shape[0], 0 : img1.shape[1]] = img1

# Find final boundary
max_x, max_y, min_y = find_boundary(final_output)
final_output = final_output[min_y:max_y, :max_x]

# Save output
cv2.imwrite("output.jpg", final_output)
print("Output saved to output.jpg")
