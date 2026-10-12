// The steps between opening a recording and asking it questions: is it compatible, are the models downloaded, and
// generating its memory (with progress, cancel and resume). Each state ends in one obvious next step.
import { useState } from "react"
import { Alert, Button, Card, Column, Field, Icon, NumberInput, Row, Spacer, Stepper, Text } from "dimos-react"
import { api, type AppStatus, type Check, type Progress } from "../core/api.ts"
import { bytes, clock, duration, gb, plural } from "../core/format.ts"
import { Bar, ModelsProgress, Stats } from "./progress.tsx"

export const RUNNING_PHASES = new Set(["models", "post_process", "planning", "map", "embedding"])
export const isGenerating = (status: AppStatus | null) => !!status?.generation && RUNNING_PHASES.has(status.generation.phase)

const PHASE_LABEL: Record<string, string> = {
    idle: "Not started",
    models: "Downloading the models first",
    planning: "Choosing keyframes",
    post_process: "Estimating poses from the lidar (lite_record)",
    map: "Building the 3D map from the lidar",
    embedding: "Embedding keyframes",
    done: "Done",
    cancelled: "Cancelled",
    failed: "Failed",
}

/** the open recording's compatibility, green or red */
export function CheckCard({ check, onPickAnother }: { check: Check; onPickAnother: () => void }) {
    const stream = (s: Check["color"]) => (s ? `${s.name} · ${s.width}×${s.height} ${s.encoding} · ${s.count.toLocaleString()} images` : "–")
    return (
        <Card padded data-compatible={check.compatible}>
            <Column gap={12}>
                <div className={`mw-verdict ${check.compatible ? "mw-verdict--ok" : "mw-verdict--bad"}`}>
                    <Icon name={check.compatible ? "check-circle" : "x-circle"} size={22} />
                    {check.compatible ? "Compatible" : "This recording can't get a memory"}
                </div>
                {check.compatible ? (
                    <dl className="mw-check-grid">
                        <dt>color</dt>
                        <dd>{stream(check.color)}</dd>
                        <dt>lidar</dt>
                        <dd>{check.lidar ? `${check.lidar}${check.lidarFrame ? ` · ${check.lidarFrame}` : ""}` : "–"}</dd>
                        <dt>intrinsics</dt>
                        <dd>{check.colorInfo ?? "–"}</dd>
                        <dt>poses</dt>
                        <dd>{check.postProcess ? "none yet: Generate estimates them from the lidar first (lite_record post_process)" : `tf, in ${check.worldFrame ?? "–"}`}</dd>
                        <dt>length</dt>
                        <dd>{duration(check.durationS)}</dd>
                    </dl>
                ) : (
                    <>
                        <Text small muted>It's missing:</Text>
                        <ul className="mw-missing" data-missing>
                            {check.missing.map((sentence) => <li key={sentence}>{sentence}</li>)}
                        </ul>
                        <Text small muted>A memory needs color images, the camera's intrinsics, a lidar, and poses (tf to the camera and the lidar, or a lidar + IMU to estimate them from). Record with a blueprint that publishes them, or pick another recording.</Text>
                    </>
                )}
                {check.notes.length > 0 && (
                    <Column gap={2}>
                        {check.notes.map((note) => <Text small muted key={note}>{note}</Text>)}
                    </Column>
                )}
                {!check.compatible && (
                    <Row>
                        <Button variant="primary" icon="arrow-left" onClick={onPickAnother} data-action="pick-another">
                            Pick another recording
                        </Button>
                    </Row>
                )}
            </Column>
        </Card>
    )
}

function GenerationProgress({ progress, status }: { progress: Progress; status: AppStatus }) {
    if (progress.phase === "models") {
        return <ModelsProgress models={status.models} />
    }
    const fraction = progress.total > 0 ? progress.done / progress.total : null
    return (
        <>
            <div className="mw-progress-head">
                <span className="mw-percent" data-generate-percent>{fraction == null ? "…" : `${Math.floor(fraction * 100)}%`}</span>
                <span className="mw-phase" data-phase={progress.phase}>{PHASE_LABEL[progress.phase] ?? progress.phase}</span>
            </div>
            <Bar value={progress.phase === "planning" ? null : fraction} running />
            <Stats
                items={[
                    ["keyframes", progress.total ? `${progress.done.toLocaleString()} / ${progress.total.toLocaleString()}` : "–"],
                    ["speed", progress.framesPerS != null ? `${progress.framesPerS.toFixed(1)} frames/s` : "–"],
                    ["time left", clock(progress.etaS)],
                    ["elapsed", clock(progress.elapsedS)],
                    ["written", bytes(progress.writtenBytes)],
                ]}
            />
            {progress.message && <Text small muted>{progress.message}</Text>}
            {progress.skipped > 0 && <Text small muted>{`${plural(progress.skipped, "keyframe")} skipped (no camera pose near them)`}</Text>}
        </>
    )
}

export function Setup({ status, onStatus, onPickAnother }: { status: AppStatus; onStatus: (status: AppStatus) => void; onPickAnother: () => void }) {
    const [everySeconds, setEverySeconds] = useState(1)
    const [busy, setBusy] = useState(false)
    const [problem, setProblem] = useState<string | null>(null)
    const check = status.check
    const memory = status.memory
    const models = status.models
    const progress = status.generation
    const generating = isGenerating(status)
    const partial = memory?.state === "partial"

    const act = async (action: () => Promise<unknown>) => {
        setBusy(true)
        setProblem(null)
        try {
            await action()
            onStatus(await api.status())
        } catch (error) {
            setProblem(String((error as Error).message ?? error))
        } finally {
            setBusy(false)
        }
    }

    const step = !check?.compatible ? 1 : !models.ready ? 2 : 3
    const steps = [
        { label: "Recording", description: status.recording?.name },
        { label: "Compatible", ...(check && !check.compatible ? { state: "failed" as const } : {}) },
        { label: "Models", description: models.ready ? "ready" : `${gb(models.totalBytes)}, once` },
        { label: "Memory", description: partial ? `${memory?.frames ?? 0} of ${memory?.planned ?? 0} keyframes` : undefined },
        { label: "Ask" },
    ]

    return (
        <div className="mw-page">
            <Stepper steps={steps} current={step} />
            {check ? <CheckCard check={check} onPickAnother={onPickAnother} /> : <Alert tone="info">Checking the recording…</Alert>}

            {check?.compatible && !models.ready && !generating && (
                <Card padded title={`Download the models (~${(models.totalBytes / 1e9 || 5.7).toFixed(1)} GB)`} description={`Three SigLIP2 models that put image patches and words in one space, and OWLv2 for boxes. Downloaded once (every file checksum-verified) into ${models.dir}.`} data-step="models">
                    <Column gap={12}>
                        {(models.downloading || models.doneBytes > 0) && <ModelsProgress models={models} />}
                        {models.error && <Alert tone="danger" title="The download stopped">{models.error}</Alert>}
                        {!models.downloading && (
                            <Row>
                                <Button variant="primary" icon="download" loading={busy} onClick={() => act(api.downloadModels)} data-action="download-models">
                                    {models.error ? "Try the download again" : models.doneBytes > 0 ? "Resume the download" : `Download models (~${(models.totalBytes / 1e9 || 5.7).toFixed(1)} GB)`}
                                </Button>
                                <Text small muted>or press Generate below: it downloads them first.</Text>
                            </Row>
                        )}
                    </Column>
                </Card>
            )}

            {check?.compatible && (
                <Card
                    padded
                    title={generating ? "Generating the memory" : partial ? "Finish the memory" : "Generate the memory"}
                    description={
                        generating
                            ? "The lidar map and each keyframe's embeddings are written into the recording itself, so it carries its own memory."
                            : "Every N seconds of the recording becomes a keyframe: its picture and pose are embedded and written into the recording in place, with the lidar's 3D map. Nothing else changes in it."
                    }
                    data-step="generate"
                >
                    <Column gap={12}>
                        {generating && progress && <GenerationProgress progress={progress} status={status} />}
                        {!generating && progress?.phase === "cancelled" && (
                            <Alert tone="warn" title="Cancelled">{`Stopped after ${plural(progress.done, "keyframe")}. Generate again resumes where it left off.`}</Alert>
                        )}
                        {!generating && progress?.phase === "failed" && (
                            <Alert tone="danger" title="Generating failed">{progress.message || "The server didn't say why."}</Alert>
                        )}
                        {!generating && partial && progress?.phase !== "cancelled" && (
                            <Alert tone="info" title="Partly done">{`${memory?.frames ?? 0} of ${memory?.planned ?? 0} keyframes have their memory. Resume picks up from there.`}</Alert>
                        )}
                        {problem && <Alert tone="danger" title="Couldn't do that" onClose={() => setProblem(null)}>{problem}</Alert>}
                        <Row wrap gap={12}>
                            {!generating && !partial && (
                                <Field inline label="a keyframe every">
                                    <NumberInput size="sm" min={0.1} max={30} step={0.5} precision={1} unit="s" value={everySeconds} onChange={setEverySeconds} style={{ width: 100 }} data-option="everySeconds" />
                                </Field>
                            )}
                            {!generating && check.durationS > 0 && !partial && (
                                <Text small muted>{`≈ ${plural(Math.max(1, Math.floor(check.durationS / everySeconds)), "keyframe")}`}</Text>
                            )}
                            <Spacer />
                            {generating ? (
                                <Button variant="danger" icon="stop" loading={busy} onClick={() => act(api.cancel)} data-action="cancel">
                                    Cancel
                                </Button>
                            ) : (
                                <Button
                                    variant="primary"
                                    size="lg"
                                    icon="sparkle"
                                    loading={busy}
                                    onClick={() => act(() => api.generate(partial ? Number(memory?.options?.everySeconds ?? everySeconds) : everySeconds))}
                                    data-action="generate"
                                >
                                    {partial || progress?.phase === "cancelled" ? "Resume" : progress?.phase === "failed" ? "Try again" : "Generate"}
                                </Button>
                            )}
                        </Row>
                        {!models.ready && !generating && <Text small muted>{`Generate downloads the models first (${gb(models.totalBytes - models.doneBytes)} left).`}</Text>}
                    </Column>
                </Card>
            )}
        </div>
    )
}
