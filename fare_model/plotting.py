"""Animation and diagnostic-plot helpers for completed FARE simulations."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from .model import FARE


#temperature and velocity pertubations from means at that altitude
def animate_temperature_velocity(
    model: "FARE",
    save: bool = True,
    output_dir: str | Path | None = None,
    filename: str = "temperature_velocity.mp4",
    z_cut: int | None = None,
    fps: int = 20,
    frames: slice | None = None,
) -> FuncAnimation:
    """Animate potential-temperature perturbations and staggered-grid velocity vectors selected by frames."""

    #prime field
    sol = _records(model)
    if save and output_dir is None:
        raise ValueError("output_dir is required when save=True.")
    z_cut = sol.pot_s.shape[2] if z_cut is None else min(z_cut, sol.pot_s.shape[2])
    frame_indices = _frame_indices(sol.count, frames)
    count = len(frame_indices)
    temperature = sol.pot_s[frame_indices, :, :z_cut]
    u = sol.u_s[frame_indices, :, :z_cut]
    x, z = sol.x[:, :z_cut], sol.z[:, :z_cut]
    # --- Interpolate staggered w to u nodes (cell centers) ---
    # w_s[i] shape is (time, Nx, Nz), representing the top faces.
    # prepend the chopped 0th boundary node (bottom wall, w=0).
    w_padded = np.zeros((count, sol.w_s.shape[1], z_cut + 1))
    w_padded[:, :, 1:] = sol.w_s[frame_indices, :, :z_cut]
    w_interp = (w_padded[:, :, :-1] + w_padded[:, :, 1:]) / 2
    # ---------------- QUIVER SETUP ----------------
    # calculate global max speed across entire simulation for stable color/scaling
    velocity = u - np.mean(u, axis=1, keepdims=True)
    max_spd = np.max(np.sqrt(velocity**2 + w_interp**2))
    max_spd = max(max_spd, np.finfo(float).eps)
    # dynamically determine subsampling for a wide aspect ratio
    # ~35 arrows horizontal and ~12 arrows vertical
    skip_x = max(1, x.shape[0] // 35)
    skip_z = max(1, x.shape[1] // 10)
    # tuning parameter for arrow length. DECREASE this number to make arrows longer.
    # adjust this until the max speed arrow just touches the next grid point.
    fig, ax = plt.subplots(figsize=(15, 9), dpi=200)
    temperature_limit = max(5.0, float(np.max(np.abs(temperature - np.mean(temperature, axis=1, keepdims=True)))))
    norm_temperature = colors.Normalize(vmin=-temperature_limit, vmax=temperature_limit)
    norm_velocity = colors.Normalize(vmin=0, vmax=max_spd)
    terrain_x, terrain_z = _terrain(model)

    def draw(frame: int) -> None:
        """Draw one temperature-and-velocity animation frame."""

        # Filled and black line contours
        ax.clear()
        theta = temperature[frame] - np.mean(temperature[frame], axis=0, keepdims=True)
        u_frame = u[frame] - np.mean(u[frame], axis=0, keepdims=True)
        w_frame = w_interp[frame]
        speed = np.sqrt(u_frame**2 + w_frame**2)
        # --- Cap max arrow length to prevent overlapping ---
        capped = np.minimum(speed, 0.4 * max_spd)
        ax.contourf(x, z, theta, levels=np.linspace(-temperature_limit, temperature_limit, 25), cmap="Spectral_r", norm=norm_temperature, extend="both")
        ax.quiver(x[::skip_x, ::skip_z], z[::skip_x, ::skip_z], (u_frame / (speed + 1e-12) * capped)[::skip_x, ::skip_z], (w_frame / (speed + 1e-12) * capped)[::skip_x, ::skip_z], speed[::skip_x, ::skip_z], cmap="Greys", norm=norm_velocity, scale=max_spd * 15, width=0.002, headwidth=4, headlength=4, headaxislength=3, minlength=0)
        if terrain_x is not None:
            ax.plot(terrain_x, terrain_z, color="black", linewidth=2, zorder=20)
        # Labels and formatting
        _style_axes(ax, x, z, f"Temperature perturbation and velocity at t = {sol.t_s[frame_indices[frame]] / 3600:.2f} hr")

    draw(0)
    # Temperature Colorbar
    temperature_map = cm.ScalarMappable(norm=norm_temperature, cmap="Spectral_r")
    # Velocity Colorbar
    velocity_map = cm.ScalarMappable(norm=norm_velocity, cmap="Greys")
    fig.colorbar(temperature_map, ax=ax, shrink=0.3, aspect=6, pad=0.02, label=r"$\theta'$ [K]")
    fig.colorbar(velocity_map, ax=ax, shrink=0.3, aspect=6, pad=0.06, label=r"$|\mathbf{v}|$ [m/s]")
    # Create and save animation
    animation = FuncAnimation(fig, draw, frames=count, interval=1000 / fps, blit=False)
    if save:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        animation.save(output / filename, writer="ffmpeg", fps=fps)
    return animation


# rain water + vapor water perturbation
def animate_rainvapor(
    model: "FARE",
    save: bool = True,
    output_dir: str | Path | None = None,
    filename: str = "rain_vapor.mp4",
    z_cut: int | None = None,
    fps: int = 20,
    frames: slice | None = None,
    vapor_limit: float = 0.004,
    rain_limit: float = 0.004,
    white_key_start: float = 0.02,
    white_key_end: float = 0.35,
) -> Path | None:
    """Render a slice-selected vapor/rain animation with a soft white-keyed rain overlay."""

    sol = _records(model)
    if save and output_dir is None:
        raise ValueError("output_dir is required when save=True.")
    if not 0 <= white_key_start < white_key_end <= 1:
        raise ValueError("white_key_start and white_key_end must satisfy 0 <= start < end <= 1.")
    z_cut = sol.qv_s.shape[2] if z_cut is None else min(z_cut, sol.qv_s.shape[2])
    frame_indices = _frame_indices(sol.count, frames)
    output = Path(output_dir) if save else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="fare_rainvapor_") as temporary_directory:
        temporary = Path(temporary_directory)
        vapor_source = temporary / "vapor_source.mp4"
        rain_source = temporary / "rain_source.mp4"
        _render_vapor_source(model, vapor_source, z_cut, frame_indices, fps, vapor_limit)
        rain_plot_box, rain_colorbar_box = _render_rain_source(model, rain_source, z_cut, frame_indices, fps, rain_limit)
        combined_path = (output / filename) if output is not None else temporary / filename
        _composite_rain_over_vapor(vapor_source, rain_source, combined_path, rain_plot_box, rain_colorbar_box, white_key_start, white_key_end, fps)
        return combined_path if save else None


def _render_vapor_source(model: "FARE", path: Path, z_cut: int, frame_indices: np.ndarray, fps: int, vapor_limit: float) -> None:
    """Render the vapor-anomaly background movie with an intentionally reserved rain-colorbar region."""

    sol = model.sol
    x, z = sol.x[:, :z_cut], sol.z[:, :z_cut]
    vapor = sol.qv_s[frame_indices, :, :z_cut]
    time = sol.t_s[frame_indices]
    count = len(frame_indices)
    terrain_x, terrain_z = _terrain(model)
    fig = plt.figure(figsize=(15, 9), dpi=200, facecolor="white")
    ax = fig.add_axes((0.07, 0.10, 0.58, 0.80))
    vapor_colorbar_axis = fig.add_axes((0.72, 0.24, 0.025, 0.52))
    norm = colors.Normalize(vmin=-vapor_limit, vmax=vapor_limit)
    vapor_map = cm.ScalarMappable(norm=norm, cmap="RdGy")
    colorbar = fig.colorbar(vapor_map, cax=vapor_colorbar_axis)
    colorbar.set_label(r"$q_v'$ [g/kg]")
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1000:.1f}"))

    def draw(frame: int) -> None:
        """Draw one vapor-anomaly source frame."""

        ax.clear()
        anomaly = vapor[frame] - np.mean(vapor[frame], axis=0, keepdims=True)
        ax.contourf(x, z, anomaly, levels=np.linspace(-vapor_limit, vapor_limit, 25), cmap="RdGy", norm=norm, extend="both")
        if terrain_x is not None:
            ax.plot(terrain_x, terrain_z, color="black", linewidth=2, zorder=20)
        _style_axes(ax, x, z, f"Vapor-water perturbation at t = {time[frame] / 3600:.2f} hr")

    draw(0)
    animation = FuncAnimation(fig, draw, frames=count, interval=1000 / fps, blit=False)
    animation.save(path, writer="ffmpeg", fps=fps)
    plt.close(fig)


def _render_rain_source(model: "FARE", path: Path, z_cut: int, frame_indices: np.ndarray, fps: int, rain_limit: float) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Render a rain-only source movie and return exact crop boxes for compositing."""

    sol = model.sol
    x, z = sol.x[:, :z_cut], sol.z[:, :z_cut]
    rain = sol.qr_s[frame_indices, :, :z_cut]
    count = len(frame_indices)
    fig = plt.figure(figsize=(15, 9), dpi=200, facecolor="white")
    ax = fig.add_axes((0.07, 0.10, 0.58, 0.80))
    rain_colorbar_axis = fig.add_axes((0.85, 0.24, 0.025, 0.52))
    norm = colors.Normalize(vmin=0, vmax=rain_limit)
    rain_map = cm.ScalarMappable(norm=norm, cmap="Blues")
    colorbar = fig.colorbar(rain_map, cax=rain_colorbar_axis)
    colorbar.set_label(r"$q_r$ [g/kg]")
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1000:.1f}"))

    def draw(frame: int) -> None:
        """Draw one rain source frame without decorations in the keyed plot region."""

        ax.clear()
        ax.set_axis_off()
        ax.contourf(x, z, rain[frame], levels=np.linspace(0, rain_limit, 25), cmap="Blues", norm=norm, extend="both")
        ax.set_aspect(6, anchor="C")

    draw(0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    bounds = ax.get_window_extent().bounds
    rain_plot_box = _matplotlib_bounds_to_cv2(bounds, height)
    rain_colorbar_box = _normalized_box_to_cv2((0.81, 0.10, 0.18, 0.80), width, height)
    animation = FuncAnimation(fig, draw, frames=count, interval=1000 / fps, blit=False)
    animation.save(path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    return rain_plot_box, rain_colorbar_box


def _composite_rain_over_vapor(vapor_path: Path, rain_path: Path, output_path: Path, rain_plot_box: tuple[int, int, int, int], rain_colorbar_box: tuple[int, int, int, int], white_key_start: float, white_key_end: float, fps: int) -> None:
    """Soft-key the rain plot from white and composite it over each vapor frame with CV2."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("animate_rainvapor requires opencv-python. Install the package dependencies before rendering.") from error

    vapor_capture = cv2.VideoCapture(str(vapor_path))
    rain_capture = cv2.VideoCapture(str(rain_path))
    width = int(vapor_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vapor_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rain_width = int(rain_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    rain_height = int(rain_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (rain_width, rain_height):
        vapor_capture.release()
        rain_capture.release()
        raise RuntimeError("The vapor and rain source videos must have identical frame dimensions.")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        vapor_capture.release()
        rain_capture.release()
        raise RuntimeError("CV2 could not open the requested animation output file.")
    plot_x, plot_y, plot_width, plot_height = rain_plot_box
    colorbar_x, colorbar_y, colorbar_width, colorbar_height = rain_colorbar_box
    try:
        while True:
            vapor_ok, vapor_frame = vapor_capture.read()
            rain_ok, rain_frame = rain_capture.read()
            if not vapor_ok and not rain_ok:
                break
            if not vapor_ok or not rain_ok:
                raise RuntimeError("The vapor and rain source videos contain different numbers of frames.")
            rain_plot = rain_frame[plot_y : plot_y + plot_height, plot_x : plot_x + plot_width].astype(np.float32)
            vapor_plot = vapor_frame[plot_y : plot_y + plot_height, plot_x : plot_x + plot_width].astype(np.float32)
            whiteness_distance = np.linalg.norm(255.0 - rain_plot, axis=2) / np.sqrt(3 * 255.0**2)
            alpha = np.clip((whiteness_distance - white_key_start) / (white_key_end - white_key_start), 0.0, 1.0)[..., None]
            vapor_frame[plot_y : plot_y + plot_height, plot_x : plot_x + plot_width] = np.rint(alpha * rain_plot + (1 - alpha) * vapor_plot).astype(np.uint8)
            vapor_frame[colorbar_y : colorbar_y + colorbar_height, colorbar_x : colorbar_x + colorbar_width] = rain_frame[colorbar_y : colorbar_y + colorbar_height, colorbar_x : colorbar_x + colorbar_width]
            writer.write(vapor_frame)
    finally:
        vapor_capture.release()
        rain_capture.release()
        writer.release()


def _matplotlib_bounds_to_cv2(bounds: tuple[float, float, float, float], height: int) -> tuple[int, int, int, int]:
    """Convert a Matplotlib display-coordinate rectangle to a CV2 top-left crop box."""

    x, y, width, box_height = bounds
    left = round(x)
    top = round(height - y - box_height)
    return left, top, round(width), round(box_height)


def _normalized_box_to_cv2(bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a Matplotlib normalized figure rectangle to a CV2 top-left crop box."""

    left, bottom, box_width, box_height = bounds
    x = round(left * width)
    y = round((1 - bottom - box_height) * height)
    return x, y, round(box_width * width), round(box_height * height)


# Conservation / Stability Plots
def plot_conservation_stability(
    model: "FARE",
    save: bool = True,
    output_dir: str | Path | None = None,
    filename: str = "conservation_stability.png",
) -> plt.Figure:
    """Plot saved temperature, water, and CFL diagnostics and optionally write the figure to output_dir."""

    sol = _records(model)
    if save and output_dir is None:
        raise ValueError("output_dir is required when save=True.")
    stats = model.stats
    count = min(sol.count, stats.count)
    time = sol.t_s[:count] / 3600
    area = model.numerics.L[0] * model.numerics.L[1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=200)
    axes[0].plot(time, stats.cons_pot[:count] / area, color="red", label="Potential Temperature")
    axes[0].plot(time, stats.cons_pote[:count] / area, color="green", label="Equivalent Potential Temperature")
    axes[0].plot(time, stats.cons_potr[:count] / area, color="blue", label="Rainy Potential Temperature")
    axes[0].set_ylabel("Average temperature [K]")
    axes[0].set_title("Temperature")
    axes[1].plot(time, stats.cons_qt[:count] * 1000 / area, color="skyblue", label="Total Water")
    axes[1].plot(time, stats.cons_qr[:count] * 1000 / area, color="blue", label="Liquid")
    axes[1].plot(time, stats.cons_qv[:count] * 1000 / area, color="grey", label="Vapor")
    axes[1].set_ylabel("Average water [g/kg]")
    axes[1].set_title("Water Content")
    axes[2].plot(time, stats.C[:count], color="lime", label="Max Courant Number")
    normalized_w = stats.w_max[:count] / max(float(np.max(stats.w_max[:count])), np.finfo(float).eps)
    axes[2].plot(time, normalized_w, color="red", label="Max Vertical Velocity (normalized)")
    axes[2].set_ylabel("Normalized magnitude")
    axes[2].set_title("CFL Stability")
    for axis in axes:
        axis.set_xlabel("Time [hr]")
        axis.minorticks_on()
        axis.grid(True, which="major", linestyle="-", linewidth=0.75)
        axis.grid(True, which="minor", linestyle=":", linewidth=0.5, color="gray")
        axis.legend()
    fig.suptitle(model.scenario.name.replace("_", " ").title())
    fig.tight_layout()
    if save:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / filename, dpi=200)
    return fig


def _frame_indices(count: int, frames: slice | None) -> np.ndarray:
    """Return the saved-frame indices selected by a slice, requiring at least one frame."""

    if frames is not None and not isinstance(frames, slice):
        raise TypeError("frames must be a slice, such as slice(None, 1600, 5), or None.")
    indices = np.arange(count)[slice(None) if frames is None else frames]
    if len(indices) == 0:
        raise ValueError("frames must select at least one saved solution.")
    return indices


def _records(model: "FARE") -> object:
    """Return populated solution storage or raise an error that explains the required solver step."""

    model._require_ready()
    if model.sol.count == 0:
        raise RuntimeError("Run solve or save an initial solution before plotting.")
    return model.sol


def _terrain(model: "FARE") -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return terrain coordinates for plotting when the active scenario supplies a height profile."""

    if model.scenario.terrain_height is None:
        return None, None
    x = model.sol.x[:, 0]
    return x, np.asarray(model.scenario.terrain_height(x))


def _style_axes(axis: plt.Axes, x: np.ndarray, z: np.ndarray, title: str) -> None:
    """Apply the notebook's kilometer-based axes, grid, and aspect styling to one animation frame."""

    axis.set_title(title)
    axis.set_xlabel("x [km]")
    axis.set_ylabel("z [km]")
    axis.set_aspect(6, anchor="C")
    axis.minorticks_on()
    axis.set_xticks(np.linspace(x.min(), x.max(), 11))
    axis.set_xticklabels(f"{value:.1f}" for value in np.linspace(x.min(), x.max(), 11) / 1000)
    axis.set_yticks(np.linspace(z.min(), z.max(), 6))
    axis.set_yticklabels(f"{value:.1f}" for value in np.linspace(z.min(), z.max(), 6) / 1000)
    axis.grid(True, which="major", linestyle="-", color="grey", linewidth=0.8, alpha=0.8)
    axis.grid(True, which="minor", linestyle="--", color="grey", linewidth=0.7, alpha=0.5)
