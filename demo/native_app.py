#!/usr/bin/env python3
"""Oncura Demo — native desktop application (double-click to open)."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from demo_core import DemoEngine


class OncuraDemoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Oncura")
        self.geometry("920x640")
        self.minsize(800, 560)

        try:
            self.engine = DemoEngine()
        except Exception as exc:
            messagebox.showerror(
                "Oncura Demo — startup error",
                f"Could not load demo models:\n\n{exc}",
            )
            self.destroy()
            return

        self.sample = None
        self._build_ui()

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=12)
        header.pack(fill="x")
        ttk.Label(header, text="Oncura", font=("Helvetica", 22, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text="Multi-modal cancer classification demo — input → classify → explain",
            font=("Helvetica", 11),
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            header,
            text="Research demonstration only. Not for clinical use.",
            font=("Helvetica", 10, "italic"),
        ).pack(anchor="w", pady=(6, 0))

        controls = ttk.LabelFrame(self, text="Workflow", padding=12)
        controls.pack(fill="x", padx=12, pady=8)

        ttk.Label(controls, text="Model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value=list(self.engine.models.keys())[0])
        ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=list(self.engine.models.keys()),
            state="readonly",
            width=28,
        ).grid(row=0, column=1, sticky="w", padx=8)

        btn_row = ttk.Frame(controls)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Button(btn_row, text="1. Load cancer sample", command=lambda: self._load_sample("cancer")).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(btn_row, text="Load control sample", command=lambda: self._load_sample("control")).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(btn_row, text="2. Classify", command=self._classify).pack(side="left")

        results = ttk.LabelFrame(self, text="Results", padding=12)
        results.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.result_text = tk.Text(results, height=5, wrap="word", font=("Menlo", 12))
        self.result_text.pack(fill="x")
        self.result_text.insert("1.0", "Load a sample, then click Classify.")
        self.result_text.configure(state="disabled")

        self.fig = Figure(figsize=(7, 3.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=results)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(12, 0))
        self._draw_chart(None)

    def _load_sample(self, kind: str) -> None:
        self.sample = self.engine.generate_sample(kind)
        label = "cancer-like" if kind == "cancer" else "control-like"
        self._set_text(f"Loaded {label} sample ({len(self.sample)} features).\nClick Classify.")

    def _classify(self) -> None:
        if self.sample is None:
            messagebox.showinfo("Oncura Demo", "Load a sample first.")
            return
        try:
            result = self.engine.predict(self.model_var.get(), self.sample)
        except Exception as exc:
            messagebox.showerror("Classification error", str(exc))
            return

        text = (
            f"Predicted type: {result['predicted_cancer_type']}\n"
            f"Confidence: {result['confidence_score']:.1%}\n"
            f"Model: {self.model_var.get()}"
        )
        self._set_text(text)
        self._draw_chart(result)

    def _set_text(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _draw_chart(self, result: dict | None) -> None:
        self.ax.clear()
        if result:
            types = result["cancer_types"]
            probs = result["class_probabilities"]
            colors = ["#2563eb" if t == result["predicted_cancer_type"] else "#94a3b8" for t in types]
            self.ax.barh(types, probs, color=colors)
            self.ax.set_xlim(0, 1)
            self.ax.set_xlabel("Probability")
            self.ax.set_title("Cancer type probabilities")
        else:
            self.ax.text(0.5, 0.5, "Probabilities will appear here", ha="center", va="center")
            self.ax.set_axis_off()
        self.fig.tight_layout()
        self.canvas.draw()


def main() -> None:
    app = OncuraDemoApp()
    if app.winfo_exists():
        app.mainloop()


if __name__ == "__main__":
    main()
