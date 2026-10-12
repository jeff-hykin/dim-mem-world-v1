// The big progress bar, its stat tiles, and the model download's progress (overall and per file).
import type { ReactNode } from "react"
import { Badge, Progress } from "dimos-react"
import type { Models, ModelFile } from "../core/api.ts"
import { clock, gb, rate, bytes } from "../core/format.ts"

export function Bar({ value, tone, running }: { value: number | null; tone?: "danger" | "warn" | "ok"; running?: boolean }) {
    const known = value != null && Number.isFinite(value)
    const width = known ? `${Math.max(0, Math.min(1, value)) * 100}%` : undefined
    const classes = ["mw-bar", running && "mw-bar--running", !known && "mw-bar--indeterminate", tone && `mw-bar--${tone}`].filter(Boolean).join(" ")
    return (
        <div className={classes} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={known ? Math.round(value * 100) : undefined}>
            <div className="mw-bar-fill" style={{ width }} />
        </div>
    )
}

export function Stats({ items }: { items: [string, ReactNode][] }) {
    return (
        <div className="mw-stats">
            {items.map(([label, value]) => (
                <div className="mw-stat" key={label}>
                    <span className="mw-stat-label">{label}</span>
                    <span className="mw-stat-value">{value}</span>
                </div>
            ))}
        </div>
    )
}

const FILE_TONE: Record<ModelFile["state"], "default" | "info" | "ok" | "warn" | "danger"> = {
    missing: "default",
    downloading: "info",
    verifying: "warn",
    ready: "ok",
    failed: "danger",
}

export function ModelsProgress({ models }: { models: Models }) {
    const fraction = models.totalBytes > 0 ? models.doneBytes / models.totalBytes : null
    return (
        <>
            <div className="mw-progress-head">
                <span className="mw-percent" data-models-percent>{fraction == null ? "…" : `${Math.floor(fraction * 100)}%`}</span>
                <span className="mw-phase">{models.ready ? "Models ready" : models.downloading ? "Downloading models" : "Models not downloaded"}</span>
            </div>
            <Bar value={fraction} running={models.downloading} tone={models.error ? "danger" : models.ready ? "ok" : undefined} />
            <Stats
                items={[
                    ["downloaded", `${gb(models.doneBytes)} / ${gb(models.totalBytes)}`],
                    ["speed", models.downloading ? rate(models.bytesPerSecond) : "–"],
                    ["time left", models.downloading ? clock(models.etaS) : "–"],
                ]}
            />
            <div className="mw-files">
                {models.files.map((file) => (
                    <div className="mw-file" key={file.name} data-model-file={file.name} title={file.error ?? `${file.model}: ${file.name}`}>
                        <span className="mw-file-name">
                            {file.name} <Badge tone={FILE_TONE[file.state]}>{file.state}</Badge>
                        </span>
                        <Progress value={file.bytes > 0 ? file.doneBytes / file.bytes : file.state === "ready" ? 1 : 0} tone={file.state === "failed" ? "danger" : file.state === "ready" ? "ok" : "primary"} />
                        <span className="mw-file-bytes">{`${bytes(file.doneBytes)} / ${bytes(file.bytes)}`}</span>
                    </div>
                ))}
            </div>
        </>
    )
}
