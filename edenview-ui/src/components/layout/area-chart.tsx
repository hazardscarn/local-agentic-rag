"use client";

// Single-series filled area chart, Task-Manager style: gridlines, a filled area
// under a thin line, current value called out in the header. Two of these (RAM,
// VRAM) stack in the sidebar rather than one combined multi-line chart -- clearer
// at this size than overlaying two series on one axis.
const WIDTH = 100;
const HEIGHT = 60;
const Y_TICKS = [100, 50, 0];

function toPath(values: number[]): { line: string; area: string } {
  if (values.length < 2) return { line: "", area: "" };
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * WIDTH;
    const y = HEIGHT - (Math.max(0, Math.min(100, v)) / 100) * HEIGHT;
    return [x, y] as const;
  });
  const line = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = `${line} L${WIDTH},${HEIGHT} L0,${HEIGHT} Z`;
  return { line, area };
}

export function AreaChart({
  label,
  values,
  currentLabel,
  colorClassName,
  gradientId,
  windowSeconds,
}: {
  label: string;
  values: number[];
  currentLabel: string;
  colorClassName: string;
  gradientId: string;
  windowSeconds: number;
}) {
  const { line, area } = toPath(values);

  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center justify-between text-[11px] text-sidebar-foreground/70">
        <span className="flex items-center gap-1.5">
          <span className={`size-2 rounded-full ${colorClassName.replace("text-", "bg-")}`} />
          {label}
        </span>
        <span className="tabular-nums">{currentLabel}</span>
      </div>
      <div className="flex gap-1.5">
        <div className="flex w-6 shrink-0 flex-col justify-between py-0.5 text-right text-[9px] leading-none text-sidebar-foreground/40 tabular-nums">
          {Y_TICKS.map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-20 w-full" preserveAspectRatio="none">
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="currentColor" stopOpacity="0.35" />
              <stop offset="100%" stopColor="currentColor" stopOpacity="0.02" />
            </linearGradient>
          </defs>
          {Y_TICKS.map((pct) => (
            <line
              key={pct}
              x1={0}
              x2={WIDTH}
              y1={HEIGHT - (pct / 100) * HEIGHT}
              y2={HEIGHT - (pct / 100) * HEIGHT}
              className="text-sidebar-border"
              stroke="currentColor"
              strokeWidth={0.5}
              strokeDasharray="2,2"
              vectorEffect="non-scaling-stroke"
            />
          ))}
          {area && <path d={area} className={colorClassName} fill={`url(#${gradientId})`} />}
          {line && (
            <path
              d={line}
              className={colorClassName}
              fill="none"
              stroke="currentColor"
              strokeWidth={1.75}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>
      </div>
      <div className="pl-8 text-right text-[9px] text-sidebar-foreground/40">last {windowSeconds}s</div>
    </div>
  );
}
