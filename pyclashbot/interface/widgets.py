from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from typing import TYPE_CHECKING

import ttkbootstrap as ttk

if TYPE_CHECKING:
    from collections.abc import Callable


class DualRingGauge(tk.Canvas):
    def __init__(
        self,
        master,
        diameter: int = 120,
        thickness: int = 12,
        start_angle: float = -90,
        fg: str = "#2ecc71",
        bg: str = "#e74c3c",
        text_color: str = "#ffffff",
        canvas_bg: str | None = None,
        **kwargs,
    ) -> None:
        size = diameter + thickness + 4
        super().__init__(master, width=size, height=size, highlightthickness=0, **kwargs)
        self.diameter = diameter
        self.thickness = thickness
        self.start_angle = start_angle
        self.foreground_colour = fg
        self.background_colour = bg
        self.text_colour = text_color
        if canvas_bg:
            self.configure(background=canvas_bg)
        self._value = 0.0
        self._arc_fg: int | None = None
        self._arc_bg: int | None = None
        self._text: int | None = None
        self._draw_static()
        self._draw_dynamic(0, "0%")

    def _bbox(self) -> tuple[int, int, int, int]:
        padding = self.thickness // 2 + 4
        return (padding, padding, padding + self.diameter, padding + self.diameter)

    def _draw_static(self) -> None:
        bbox = self._bbox()
        if self._arc_bg:
            self.delete(self._arc_bg)
        self._arc_bg = self.create_arc(
            *bbox,
            start=0,
            extent=359.9,
            style="arc",
            width=self.thickness,
            outline=self.background_colour,
        )
        centre = self.winfo_reqwidth() // 2
        if self._text:
            self.delete(self._text)
        label_size = 14 if self.diameter <= 80 else 16
        self._text = self.create_text(
            centre,
            centre,
            text="0%",
            fill=self.text_colour,
            font=("-size", label_size, "-weight", "bold"),
        )

    def _draw_dynamic(
        self,
        percent: float,
        label: str,
        fg_colour: str | None = None,
        text_colour: str | None = None,
    ) -> None:
        percent = max(0, min(100, float(percent)))
        bbox = self._bbox()
        extent = percent / 100 * 360
        outline = fg_colour or self.foreground_colour
        if self._arc_fg:
            self.delete(self._arc_fg)
        self._arc_fg = self.create_arc(
            *bbox,
            start=self.start_angle,
            extent=extent,
            style="arc",
            width=self.thickness,
            outline=outline,
        )
        if self._text is not None:
            self.itemconfig(self._text, text=label, fill=text_colour or self.text_colour)
        self._value = percent

    def animate_to(
        self,
        target_percent: float,
        label_format: str = "{:.0f}%",
        fg_colour: str | None = None,
        text_colour: str | None = None,
        duration_ms: int = 400,
    ) -> None:
        start = self._value
        target = max(0, min(100, float(target_percent)))
        steps = max(1, duration_ms // 10)
        delta = (target - start) / steps
        frame = 0

        def step() -> None:
            nonlocal frame, start
            start += delta
            frame += 1
            final_frame = frame >= steps
            current = target if final_frame else start
            self._draw_dynamic(current, label_format.format(current), fg_colour, text_colour)
            if not final_frame:
                self.after(10, step)

        step()

    def set_colours(self, fg: str, bg: str, text: str, *, canvas_bg: str | None = None) -> None:
        """Update the colours used by the gauge."""
        self.foreground_colour = fg
        self.background_colour = bg
        self.text_colour = text
        if canvas_bg:
            self.configure(background=canvas_bg)
        self._draw_static()
        self._draw_dynamic(self._value, f"{self._value:.0f}%", fg, text)


class ScrollablePage(ttk.Frame):
    """Vertically scrolling page: widgets go in ``self.body``, which always spans the canvas width.

    The scrollbar only shows while the body is taller than the visible area. Mouse-wheel routing
    lives in the window (one ``bind_all``), which calls ``scroll_units`` on the page under the pointer.
    """

    def __init__(self, master: tk.Misc, padding: int | tuple[int, ...] = 16, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.body = ttk.Frame(self.canvas, padding=padding)
        self._window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self._bar_pending = False
        self.body.bind("<Configure>", self._on_body_configure, add="+")
        self.canvas.bind("<Configure>", self._on_canvas_configure, add="+")

    def overflows(self) -> bool:
        return self.body.winfo_reqheight() > max(1, self.canvas.winfo_height())

    def scroll_units(self, units: int) -> None:
        if units and self.overflows():
            self.canvas.yview_scroll(units, "units")

    def set_background(self, colour: str) -> None:
        self.canvas.configure(background=colour)

    def _on_body_configure(self, _event: object = None) -> None:
        self.canvas.configure(scrollregion=(0, 0, self.body.winfo_reqwidth(), self.body.winfo_reqheight()))
        self._schedule_bar_sync()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)
        self._schedule_bar_sync()

    def _schedule_bar_sync(self) -> None:
        if not self._bar_pending:
            self._bar_pending = True
            self.after_idle(self._sync_bar)

    def _sync_bar(self) -> None:
        self._bar_pending = False
        try:
            if self.overflows():
                self.vbar.grid()
            else:
                self.vbar.grid_remove()
                self.canvas.yview_moveto(0)
        except tk.TclError:
            return


class ResponsiveGrid(ttk.Frame):
    """Lays child cards out in up to ``max_columns`` equal columns, dropping to fewer when narrow."""

    def __init__(
        self,
        master: tk.Misc,
        min_column_width: int = 360,
        max_columns: int = 2,
        gap: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.min_column_width = min_column_width
        self.max_columns = max_columns
        self.gap = gap
        self.columns = max_columns
        self._cards: list[tk.Widget] = []
        self.bind("<Configure>", self._on_configure, add="+")

    def add(self, card: tk.Widget) -> tk.Widget:
        self._cards.append(card)
        self._layout()
        return card

    def columns_for_width(self, width: int) -> int:
        return max(1, min(self.max_columns, (width + self.gap) // (self.min_column_width + self.gap)))

    def _on_configure(self, event: tk.Event) -> None:
        if event.widget is not self or event.width <= 1:
            return
        columns = self.columns_for_width(event.width)
        if columns != self.columns:
            self.columns = columns
            self._layout()

    def _layout(self) -> None:
        for index in range(self.max_columns):
            used = index < self.columns
            self.columnconfigure(index, weight=1 if used else 0, uniform="cards" if used else "")
        half = self.gap // 2
        for index, card in enumerate(self._cards):
            row, column = divmod(index, self.columns)
            padx = (0 if column == 0 else half, 0 if column == self.columns - 1 else half)
            card.grid(row=row, column=column, sticky="nsew", padx=padx, pady=(0, self.gap))


class NavButton(tk.Frame):
    """Sidebar entry: fixed-width glyph column + label, hover and active highlight, click runs ``command``."""

    def __init__(
        self,
        master: tk.Misc,
        text: str,
        glyph: str,
        command: Callable[[], None],
        font: tkfont.Font | str = "TkDefaultFont",
        glyph_width: int = 3,
    ) -> None:
        super().__init__(master, cursor="hand2", padx=8, pady=7)
        self.command = command
        self.active = False
        self._hover = False
        self._colours = {
            "bg": "#2e2e2e",
            "hover": "#3a3a3a",
            "active": "#375a7f",
            "fg": "#ffffff",
            "active_fg": "#ffffff",
        }
        self.glyph_label = tk.Label(self, text=glyph, width=glyph_width, anchor="center", font=font)
        self.glyph_label.pack(side="left")
        self.text_label = tk.Label(self, text=text, anchor="w", font=font)
        self.text_label.pack(side="left", fill="x", expand=True, padx=(4, 12))
        for widget in (self, self.glyph_label, self.text_label):
            widget.bind("<Button-1>", self._on_click, add="+")
            widget.bind("<Enter>", self._on_enter, add="+")
            widget.bind("<Leave>", self._on_leave, add="+")
        self._paint()

    def set_colours(self, *, bg: str, hover: str, active: str, fg: str, active_fg: str) -> None:
        self._colours = {"bg": bg, "hover": hover, "active": active, "fg": fg, "active_fg": active_fg}
        self._paint()

    def set_active(self, active: bool) -> None:
        self.active = active
        self._paint()

    def _on_click(self, _event: object = None) -> None:
        self.command()

    def _on_enter(self, _event: object = None) -> None:
        self._hover = True
        self._paint()

    def _on_leave(self, _event: object = None) -> None:
        self._hover = False
        self._paint()

    def _paint(self) -> None:
        c = self._colours
        if self.active:
            bg, fg = c["active"], c["active_fg"]
        else:
            bg, fg = (c["hover"] if self._hover else c["bg"]), c["fg"]
        self.configure(bg=bg)
        for label in (self.glyph_label, self.text_label):
            label.configure(bg=bg, fg=fg)
