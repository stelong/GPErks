import numpy as np


def get_col(color_name=None):
    """Material Design color palettes (only '100' and '900' variants).
    Help: call with no arguments to see the list of available colors, these are also returned into a list
    Kwarg:
            - color_name: string representing the color's name
    Output:
            - color: list of two elements
                    [0] = lightest color '100'-variant (RGB-triplet in [0, 1])
                    [1] = darkest color '900'-variant (RGB-triplet in [0, 1])
    """
    colors = {
        "red": [[255, 205, 210], [183, 28, 28]],
        "pink": [[248, 187, 208], [136, 14, 79]],
        "purple": [[225, 190, 231], [74, 20, 140]],
        "deep_purple": [[209, 196, 233], [49, 27, 146]],
        "indigo": [[197, 202, 233], [26, 35, 126]],
        "blue": [[187, 222, 251], [13, 71, 161]],
        "light_blue": [[179, 229, 252], [1, 87, 155]],
        "cyan": [[178, 235, 242], [0, 96, 100]],
        "teal": [[178, 223, 219], [0, 77, 64]],
        "green": [[200, 230, 201], [27, 94, 32]],
        "light_green": [[220, 237, 200], [51, 105, 30]],
        "lime": [[240, 244, 195], [130, 119, 23]],
        "yellow": [[255, 249, 196], [245, 127, 23]],
        "amber": [[255, 236, 179], [255, 111, 0]],
        "orange": [[255, 224, 178], [230, 81, 0]],
        "deep_orange": [[255, 204, 188], [191, 54, 12]],
        "brown": [[215, 204, 200], [62, 39, 35]],
        "gray": [[245, 245, 245], [33, 33, 33]],
        "blue_gray": [[207, 216, 220], [38, 50, 56]],
    }
    if not color_name:
        print("\n=== Colors available are:")
        for key, _ in colors.items():
            print("- " + key)
        return list(colors.keys())
    else:
        color = [
            [colors[color_name][i][j] / 255 for j in range(3)]
            for i in range(2)
        ]
        return color


def interp_col(color, n):
    """Linearly interpolate a color.
    Args:
            - color: list with two elements:
                    color[0] = lightest color variant (get_col('color_name')[0])
                    color[1] = darkest color variant (get_col('color_name')[1]).
            - n: number of desired output colors (n >= 2).
    Output:
            - lsc: list of n linearly scaled colors.
    """
    c = [
        np.interp(list(range(1, n + 1)), [1, n], [color[0][i], color[1][i]])
        for i in range(3)
    ]
    lsc = [[c[0][i], c[1][i], c[2][i]] for i in range(n)]
    return lsc
