class ColorChoice:
    def __init__(
        self,
        security_name: str,
        dark_line_color: str,
        light_markers_color: str,
        dark_color: str,
        marker: str,
        alpha: float = None,
    ):
        self.security_name = security_name
        self.dark_line_color = dark_line_color
        self.light_markers_color = light_markers_color
        self.dark_color = dark_color
        self.marker = marker
        if alpha is not None:
            self.alpha = alpha


TSLA_COLORS = ColorChoice(
    security_name="TSLA",
    dark_line_color="#0E1111",
    light_markers_color="#F60E1A",  # 3FBAFF
    dark_color="#380404",  # 0E1111
    marker="^",
    alpha=0.65,
)

MSFT_COLORS = ColorChoice(
    security_name="MSFT",
    dark_line_color="#0E1111",
    light_markers_color="#009FF8",
    dark_color="#0E1111",
    marker="o",
    alpha=1,
)

AMZN_COLORS = ColorChoice(
    security_name="AMZN",
    dark_line_color="#0E1111",
    light_markers_color="#FCC400",
    dark_color="#0E1111",
    marker="D",
    alpha=0.6,
)

NFLX_COLORS = ColorChoice(
    security_name="NFLX",
    dark_line_color="#0E1111",
    light_markers_color="#FF8624",
    dark_color="#0E1111",
    marker="s",
    alpha=0.8,
)

EBAY_COLORS = ColorChoice(
    security_name="EBAY",
    dark_line_color="#0E1111",
    light_markers_color="#00FF0C",  # 6FFF7C
    dark_color="#0E1111",  # #043E07
    marker="+",
    alpha=0.95,
)
