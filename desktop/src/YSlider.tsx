/**
 * Vertical Y-scrubber for the 3D isometric map.
 *
 * The mouse wheel is the primary Y-scrub input (one scale-Y unit per
 * notch); this slider is the secondary, direct-manipulation control —
 * useful for jumping to a specific Y like Y=15 (deep dark / deepslate)
 * or Y=192 (mountain peaks) without scrolling for ten seconds.
 *
 * Layout: thin vertical track pinned to the right edge of the 3D pane,
 * with depth-band labels (sky / surface / sea / deepslate) and a thumb
 * that shows the current block-Y. Pointer-drag the thumb or click
 * anywhere on the track to jump.
 */

import { useCallback, useRef } from "react";

export type YSliderProps = {
  y: number;
  yMin: number;
  yMax: number;
  onChange: (y: number) => void;
};

export function YSlider({ y, yMin, yMax, onChange }: YSliderProps) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef(false);

  // Block-Y → top-edge percent (top of the track = yMax, bottom = yMin).
  const yToPct = (yy: number) =>
    100 * (1 - (yy - yMin) / (yMax - yMin));

  const pctToY = useCallback(
    (pct: number) => {
      const p = Math.max(0, Math.min(100, pct));
      return Math.round(yMax - (p * (yMax - yMin)) / 100);
    },
    [yMin, yMax],
  );

  const updateFromPointer = useCallback(
    (e: React.PointerEvent) => {
      const rect = trackRef.current?.getBoundingClientRect();
      if (!rect) return;
      const pct = ((e.clientY - rect.top) / rect.height) * 100;
      onChange(pctToY(pct));
    },
    [onChange, pctToY],
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      draggingRef.current = true;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      updateFromPointer(e);
    },
    [updateFromPointer],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!draggingRef.current) return;
      updateFromPointer(e);
    },
    [updateFromPointer],
  );

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    draggingRef.current = false;
    try {
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      /* pointer already released */
    }
  }, []);

  // A few cosmetic depth-band markers — purely visual orientation.
  const bands = [
    { y: yMax, label: "sky" },
    { y: 192, label: "peaks" },
    { y: 63, label: "sea" },
    { y: 0, label: "ground" },
    { y: -32, label: "caves" },
    { y: yMin, label: "deepslate" },
  ].filter((b) => b.y >= yMin && b.y <= yMax);

  return (
    <div
      className="ySlider"
      role="slider"
      aria-valuenow={y}
      aria-valuemin={yMin}
      aria-valuemax={yMax}
      aria-label="Y level"
    >
      <div className="ySliderLabel ySliderTop">Y {yMax}</div>
      <div
        ref={trackRef}
        className="ySliderTrack"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        {bands.map((b) => (
          <div
            key={b.label}
            className="ySliderBand"
            style={{ top: `${yToPct(b.y)}%` }}
            title={`Y=${b.y} (${b.label})`}
          >
            <span>{b.label}</span>
          </div>
        ))}
        <div
          className="ySliderThumb"
          style={{ top: `${yToPct(y)}%` }}
          aria-hidden
        >
          {y}
        </div>
      </div>
      <div className="ySliderLabel ySliderBottom">Y {yMin}</div>
    </div>
  );
}
