from typing import List
import matplotlib.pyplot as plt
import numpy as np
import csv

# Dictionary of small regions
# Dimensions are in [4, 8, 12, 14, 16, 18, 22, 30, 32, 34, 36, 38, 48, 50, 54, 58, 66, 70, 74, 84, 100].
# key: dimensions, value: (top left row, top left column, bottom right row, bottom right column)
REGIONS = {
    100: (0, 0, 99, 99),
    70: (0, 100, 69, 169),
    54: (0, 170, 53, 223),
    16: (54, 170, 69, 185),
    38: (54, 186, 91, 223),
    30: (70, 100, 99, 129),
    34: (70, 130, 103, 163),
    22: (70, 164, 91, 185),
    12: (92, 164, 103, 175),
    4: (100, 126, 103, 129),
    58: (100, 0, 157, 57),
    50: (100, 58, 149, 107),
    18: (100, 108, 117, 125),
    14: (104, 126, 117, 139),
    36: (104, 140, 139, 175),
    48: (92, 176, 139, 223),
    32: (118, 108, 149, 139),
    8: (150, 58, 157, 65),
    66: (158, 0, 223, 65),
    74: (150, 66, 223, 139),
    84: (140, 140, 223, 223),
}


MIN_NUMS = {
    4: 15347,
    8: 19629,
    12: 11029,
    14: 15355,
    16: 8909,
    18: 15185,
    22: 10787,
    30: 9759,
    32: 24577,
    34: 10209,
    36: 15453,
    38: 9037,
    48: 11101,
    50: 13935,
    54: 7451,
    58: 12253,
    66: 22399,
    70: 5001,
    74: 19661,
    84: 16101,
    100: 1,
}

SQUARED_224 = 224**2

ADJACENT_REGIONS = {
      4: [ 14,  18,  30,  34],
      8: [ 50,  58,  66,  74],
     12: [ 22,  34,  36,  48],
     14: [  4,  18,  32,  34,  36],
     16: [ 22,  38,  54,  70],
     18: [  4,  14,  30,  32,  50],
     22: [ 12,  16,  34,  38,  48,  70],
     30: [  4,  18,  34,  50,  70, 100],
     32: [ 14,  18,  36,  50,  74,  84],
     34: [  4,  12,  14,  22,  30,  36,  70],
     36: [ 12,  14,  32,  34,  48,  84],
     38: [ 16,  22,  48,  54],
     48: [ 12,  22,  36,  38,  84],
     50: [  8,  18,  30,  32,  58,  74, 100],
     54: [ 16,  38,  70],
     58: [  8,  50,  66, 100],
     66: [  8,  58,  74],
     70: [ 16,  22,  30,  34,  54, 100],
     74: [  8,  32,  50,  66,  84],
     84: [ 32,  36,  48,  74],
    100: [ 30,  50,  58,  70],
}

COLORS = [
    "#8AC926",  # Yellow-green
    "#FFCA3A",  # Bright orange
    "#FF595E",  # Reddish pink 
    "#1982C4",  # Blue
    "#D2691E",  # Chocolate
    "#FF7F50",  # Coral
    "#00BFFF",  # Deep sky blue 
    "#7FFF00",  # Chartreuse
    "#F08080",  # Light coral
    "#FFD700",  # Gold
    "#808000",  # Olive
    "#48D1CC",  # Turquoise
    "#C71585",  # Medium violet red
    "#B0E0E6",  # Powder blue
    "#CD853F",  # Peru (brownish)
    "#5F9EA0",  # Cadmium green
    "#FFDAB9",  # Peach puff
    "#9932CC",  # Dark orchid
    "#F4A460",  # Sandy brown
    "#00FA9A",  # Mint green
    "#6A4C93",  # Deep purple
]

REGIONS_AND_COLORS = [(r[0], r[1], r[2], r[3], c) for (r, c) in zip(REGIONS.values(), COLORS)]

# The locations on the outer boundary where the secondary diagonal passes (indices are 0-indexed for each small square).
# Since there are two such locations, they are represented as [(row of the first location, col of the first location), (row of the second location, col of the second location)].

MAIN_DIAG_POS = {
    18: [(8, 0), (17, 9)],
    50: [(0, 42), (7, 49)]
}

ANTI_DIAG_POS = {
    18: [(0, 15), (15, 0)],
    30: [(24, 29), (29, 24)],
    34: [(0, 23), (23, 0)],
    50: [(49, 16), (16, 49)],
    70: [(54, 69), (69, 54)],
    74: [(7, 0), (0, 7)],
}

class MagicSquare:
    def __init__(self, n: int) -> None:
        self.n = n
        self.magicsq = [[0 for _ in range(n)] for _ in range(n)]

    def sum_row(self, i: int) -> int:
        return sum(self.magicsq[i])
    
    def sum_col(self, j: int) -> int:
        return sum([self.magicsq[i][j] for i in range(self.n)])
    
    def sum_main_diag(self) -> int:
        return sum([self.magicsq[i][i] for i in range(self.n)])
    
    def sum_anti_diag(self) -> int:
        return sum([self.magicsq[i][self.n - i - 1] for i in range(self.n)])
     
    def is_valid(self) -> bool:
        total_value = sum(self.magicsq[0])
        for i in range(self.n):
            if self.sum_row(i) != total_value:
                return False
        for j in range(self.n):
            if self.sum_col(j) != total_value:
                return False
        if self.sum_main_diag() != total_value:
            return False
        if self.sum_anti_diag() != total_value:
            return False
        return True
    
    def debug_valid(self) -> None:
        total_value = sum(self.magicsq[0])
        flag = True
        for i in range(self.n):
            if self.sum_row(i) != total_value:
                flag = False
        if flag:
            print(f"✓ Row's conditions are satisfied: {total_value}")
        else:
            print("× Row's conditions are not satisfied.")
        flag = True
        for j in range(self.n):
            if self.sum_col(j) != total_value:
                flag = False
        if flag:
            print(f"✓ Col's conditions are satisfied: {total_value}")
        else:
            print("× Col's conditions are not satisfied.")
        if self.sum_main_diag() == total_value:
            print(f"✓ Main diagonal's conditions are satisfied: {total_value}")
        else:
            print(f"× Main diagonal's conditions are not satisfied: main_diag - sum = {self.sum_main_diag()} - {total_value} = {self.sum_main_diag() - total_value}")
        if self.sum_anti_diag() == total_value:
            print(f"✓ Anti diagonal's conditions are satisfied: {total_value}")
        else:
            print(f"× Anti diagonal's conditions are not satisfied: anti_diag - sum =  {self.sum_anti_diag()} - {total_value} = {self.sum_anti_diag() - total_value}")
        return True
    
    def write_csv(self, filename = "Abe_224.csv"):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.magicsq)

    def print_magicsq(self) -> None:
        # determine the maximum digit
        max_digit = 0
        for row in self.magicsq:
            for e in row:
                max_digit = max(len(str(e)), max_digit)
        max_digit += 1

        for row in self.magicsq:
            for e in row:
                print(f"{e:>{max_digit}}", end = "")
            print("")

    def assign(self, i: int, j: int, num: int) -> None:
        self.magicsq[i][j] = num

    def assign_row(self, i: int, nums: List[int]) -> None:
        for j in range(self.n):
            self.assign(i, j, nums[j])

    def assing_col(self, j: int, nums: List[int]) -> None:
        for i in range(self.n):
            self.assign(i, j, nums[i])

    def replace(self, i1: int, j1: int, i2: int, j2: int) -> None:
        self.magicsq[i1][j1], self.magicsq[i2][j2] = self.magicsq[i2][j2], self.magicsq[i1][j1]

    def search(self, num: int) -> tuple[int, int]:
        for i in range(self.n):
            for j in range(self.n):
                if self.magicsq[i][j] == num:
                    return (i, j)
        raise ValueError(f"{num} is not found in the magic square")
    
    def search_and_replace(self, num1: int, num2: int) -> None:
        i1, j1 = self.search(num1)
        i2, j2 = self.search(num2)
        self.replace(i1, j1, i2, j2)

    def rotate(self) -> None:
        # rotate anti-clockwise
        ms = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                ms[self.n-1-j][i] = self.magicsq[i][j]
        self.magicsq = ms

    def draw_pic(self, filename: str = "magic_square.png", color_list = None, dpi = 100) -> None:
        """
        Function to draw and save the magic square

        Parameters
        ----------
        filename : str, optional
            filename, by default "magic_square.png"
        color_list : list, optional
           (top left row, top left column, bottom right row, bottom right column, color code), by default None
        dpi : int, optional
            dpi, by default 100
        """
        matrix = np.array(self.magicsq)

        fig, ax = plt.subplots(figsize=(self.n, self.n))
        ax.set_xticks(np.arange(self.n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="both", size=0, labelbottom=False, labelleft=False)
        
        # color the background
        if color_list is not None:
            for row_start, col_start, row_end, col_end, color in color_list:
                for i in range(row_start, row_end + 1):
                    for j in range(col_start, col_end + 1):
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color, alpha=0.5))

        for i in range(self.n):
            for j in range(self.n):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=12, fontweight="bold")
        
        ax.set_xlim(-0.5, self.n - 0.5)
        ax.set_ylim(self.n - 0.5, -0.5)
        
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close(fig)


def Abe224() -> MagicSquare:
    # 224×224 magic square made by Abe in 1991.
    # G. Abe, Suri Kagaku, 29-12, 1991, pp. 64-68 in Japanese.

    def fill_regions() -> MagicSquare:
        ms = MagicSquare(224)
        for k, v in MIN_NUMS.items():
            small_ms = None
            if k == 4:
                small_ms = pattern_4(v)
            elif k % 8 == 0:
                small_ms = pattern_8m(k, v)
            elif k % 8 == 4:
                small_ms = pattern_8m_4(k, v)
            else:
                small_ms = pattern_8m_2_8m_6(k, v)
            row_0, col_0, _, _ = REGIONS[k]
            for i in range(k):
                for j in range(k):
                    ms.assign(row_0 + i, col_0 + j, small_ms.magicsq[i][j])
        return ms
    
    def check_valid_in_small_regions(ms) -> bool:
        is_valid = True
        for k, r in REGIONS.items():
            small_ms = MagicSquare(k)
            for i in range(k):
                for j in range(k):
                    small_ms.magicsq[i][j] = ms.magicsq[r[0] + i][r[1] + j]
            is_valid *= small_ms.is_valid()

        return is_valid
    
    def debug_valid(ms) -> None:
        ms.debug_valid()
        if count_num(ms):
            print("✓ All numbers are different")
        else:
            print("× There are same numbers")
        if check_valid_in_small_regions(ms):
            print("✓ All small regions are valid")
        else:
            print("× There are invalid small regions")

    def count_num(ms) -> int:
        set_num = set()
        set_num = set([ms.magicsq[i][j] for i in range(ms.n) for j in range(ms.n)])
        return (len(set_num) == ms.n ** 2)
    
    def search_and_replace_with_complement(ms, num1, num2) -> None:
        i1, j1 = ms.search(num1)
        i2, j2 = ms.search(num2)
        ms.replace(i1, j1, i2, j2)
        i3, j3 = ms.search(SQUARED_224 + 1 - num1)
        i4, j4 = ms.search(SQUARED_224 + 1 - num2)
        ms.replace(i3, j3, i4, j4)

    ms = fill_regions()

    # Swap to satisfy the condition of main diagonal
    search_and_replace_with_complement(ms, 15199, 15200)  # in 18
    search_and_replace_with_complement(ms, 34967, 34986)  # in 18
    search_and_replace_with_complement(ms, 36186, 36237)  # in 50

    # Swap to satisfy the condition of anti diagonal
    search_and_replace_with_complement(ms, 15205, 34974)  # in 18
    search_and_replace_with_complement(ms,  9810, 40365)  # in 30
    search_and_replace_with_complement(ms, 10214, 39912)  # in 34
    search_and_replace_with_complement(ms, 14000, 36145)  # in 50
    search_and_replace_with_complement(ms,  5007, 45053)  # in 70
    search_and_replace_with_complement(ms, 19673, 30456)  # in 74

    debug_valid(ms)
    return ms

# A magic square with a side length of 8m
def pattern_8m(n: int, min_num: int) -> MagicSquare:
    ms = MagicSquare(n)
    half_n = int(n/2)
    n_square = n * n
    half_of_n_square = int(n_square / 2)
    row = 0
    col = n-1
    diff = -1
    count = 0
    for i in range(min_num, min_num + half_of_n_square):
        ms.assign(row, col, i)
        col += diff
        count += 1
        if count % half_n == 0:
            row += 1
        if count % n == 0:
            diff*= -1
            col += diff

    row = n-1
    col = 0
    diff = 1
    count = 0
    for i in range(SQUARED_224 + 1 - min_num - half_of_n_square + 1, SQUARED_224 + 1 - min_num + 1):
        ms.assign(row, col, i)
        col += diff
        count += 1
        if count % half_n == 0:
            row -= 1
        if count % n == 0:
            diff*= -1
            col += diff

    replace_list = []
    replace_list.append((half_n-1, half_n, half_n-1, n-1))
    replace_list.append((half_n, half_n, half_n, n-1))
    replace_list.append((half_n-1, half_n, half_n, half_n))
    replace_list.append((half_n-1, 0, half_n-1, half_n-1))
    replace_list.append((half_n, 0, half_n, half_n-1))
    replace_list.append((half_n-1, half_n-1, half_n, half_n-1))
    for replace in replace_list:
        ms.replace(*replace)

    return ms

# A magic square with a side length of 4
def pattern_4(min_num: int) -> MagicSquare:
    ms = MagicSquare(4)
    comp = SQUARED_224 + 1 - min_num
    ms.assign(0, 0, min_num)
    ms.assign(2, 1, min_num + 1)
    ms.assign(1, 2, min_num + 2)
    ms.assign(3, 3, min_num + 3)
    ms.assign(3, 2, min_num + 4)
    ms.assign(1, 3, min_num + 5)
    ms.assign(2, 0, min_num + 6)
    ms.assign(0, 1, min_num + 7)
    ms.assign(2, 2, comp)
    ms.assign(0, 3, comp - 1)
    ms.assign(3, 0, comp - 2)
    ms.assign(1, 1, comp - 3)
    ms.assign(1, 0, comp - 4)
    ms.assign(3, 1, comp - 5)
    ms.assign(0, 2, comp - 6)
    ms.assign(2, 3, comp - 7)
    return ms

# A magic square with a side length of 8m+4, m>= 1
def pattern_8m_4(n: int, min_num: int) -> MagicSquare:
    ms = MagicSquare(n)
    half_n = int(n/2)
    n_square = n * n
    half_of_n_square = int(n_square / 2)
    row = 0
    col = n-1
    diff = -1
    count = 0
    for i in range(min_num, min_num + half_of_n_square):
        ms.assign(row, col, i)
        col += diff
        count += 1
        if count % half_n == 0:
            row += 1
        if count % n == 0:
            diff*= -1
            col += diff

    row = n-1
    col = 0
    diff = 1
    count = 0
    for i in range(SQUARED_224 + 1 - min_num - half_of_n_square + 1, SQUARED_224 + 1 - min_num + 1):
        ms.assign(row, col, i)
        col += diff
        count += 1
        if count % half_n == 0:
            row -= 1
        if count % n == 0:
            diff*= -1
            col += diff

    replace_list = []
    replace_list.append((half_n-1, half_n-1, half_n-2, n-1))
    replace_list.append((half_n-1, half_n, half_n-2, 0))
    replace_list.append((half_n, half_n-1, half_n+1, n-1))
    replace_list.append((half_n, half_n, half_n+1, 0))
    for replace in replace_list:
        ms.replace(*replace)

    return ms

# # A magic square with a side length of 8m+2 or 8m+6
def pattern_8m_2_8m_6(n: int, min_num: int) -> MagicSquare:

    def assign_with_complement(assign_list, i, j, num):
        assign_list.append((i, j, num))
        complement_num = SQUARED_224 + 1 - num
        if ((i == 0) and (j == 0)) or ((i==n-1) and (j==n-1)) or ((i==0) and (j==n-1)) or ((i==n-1) and (j==0)):
            assign_list.append((n-1-i, n-1-j, complement_num))
        elif (i==0) or (i==n-1):
            assign_list.append((n-1-i, j, complement_num))
        elif (j==0) or (j==n-1):
            assign_list.append((i, n-1-j, complement_num))


    ms = MagicSquare(n)
    num_outer_frame = 4*(n - 1)
    half_num_outer_frame = num_outer_frame // 2
    inner_ms = None
    inner_min_num = min_num + half_num_outer_frame
    if n % 8 == 2:
        inner_ms = pattern_8m(n-2, inner_min_num)
    elif n % 8 == 6:
        inner_ms = pattern_8m_4(n-2, inner_min_num)
    else:
        raise ValueError("n must be 8m+2 or 8m+6")
    for i in range(n-2):
        for j in range(n-2):
            ms.assign(i+1, j+1, inner_ms.magicsq[i][j])

    assign_list = []
    assign_with_complement(assign_list, 0, 0, min_num)
    assign_with_complement(assign_list, 0, n-1, min_num + 1)
    assign_with_complement(assign_list, n-1, 4, min_num + 2)
    assign_with_complement(assign_list, n-1, 3, min_num + 3)
    assign_with_complement(assign_list, n-1, 2, min_num + 4)
    assign_with_complement(assign_list, 1, 0, min_num + 5)
    assign_with_complement(assign_list, 2, n-1, min_num + 6)
    assign_with_complement(assign_list, 3, n-1, min_num + 7)
    assign_with_complement(assign_list, 0, 1, min_num + 8)
    assign_with_complement(assign_list, 4, 0, min_num + 9)

    num = min_num + 10
    num_remainder = n - 6
    half_num_remainder = num_remainder // 2
    col = 5
    for i in range(num_remainder):
        row = n-1 if (half_num_remainder // 2 <= i < half_num_remainder + half_num_remainder // 2) else 0
        assign_with_complement(assign_list, row, 5 + i, num)
        num += 1
    row = 5
    for i in range(num_remainder):
        col = n-1 if (half_num_remainder // 2 <= i < half_num_remainder + half_num_remainder // 2) else 0
        assign_with_complement(assign_list, 5 + i, col, num)
        num += 1
    
    for i, j, num in assign_list:
        ms.assign(i, j, num)

    return ms
