// Ask the memory: a question box over the 3D scene (the memory's point cloud and the camera's path), each answer's
// places drawn as boxes the camera flies to, and a panel of the places with their best sighting as evidence.
import { useEffect, useRef, useState, type FormEvent } from "react"
import { Alert, Button, Chip, Column, EmptyState, Hud, Icon, IconButton, NumberInput, Progress, Row, Spinner, Text } from "dimos-react"
import { Scene3D, type Viewer } from "dimos-react/three"
import { api, type Answer, type AppStatus, type Place } from "../core/api.ts"
import { meters, plural } from "../core/format.ts"
import { MemoryScene } from "../scene/memory_scene.ts"

const EXAMPLES = ["traffic cone", "chair", "door", "person", "plant", "trash can"]

/** a sighting's time: on the recording's clock when it's inside it, otherwise the wall clock */
function seenAt(ts: number, status: AppStatus): string {
    const check = status.check
    if (check && ts >= check.start - 1 && ts <= check.end + 1) {
        const rel = Math.max(0, ts - check.start)
        return `${Math.floor(rel / 60)}:${String(Math.floor(rel % 60)).padStart(2, "0")}`
    }
    const date = new Date(ts > 1e12 ? ts : ts * 1000)
    return Number.isFinite(date.getTime()) ? date.toLocaleTimeString() : String(ts)
}

const answerKey = (answer: Answer | null) => (answer ? `${answer.query}|${answer.timings?.totalMs}|${answer.count}` : "")

function PlaceCard({ place, index, query, selected, status, onClick }: { place: Place; index: number; query: string; selected: boolean; status: AppStatus; onClick: () => void }) {
    const [broken, setBroken] = useState(false)
    const [x, y, z] = place.center
    const [sx, sy, sz] = place.size
    return (
        <button type="button" className={`mw-place${selected ? " mw-place--selected" : ""}`} onClick={onClick} data-place={place.id}>
            <div className="mw-thumb">
                {!broken && <img src={api.evidenceUrl(place.id, query)} alt={`#${index + 1}: the best sighting of ${query}`} loading="lazy" onError={() => setBroken(true)} />}
                {broken && <div className="mw-thumb-missing">no picture</div>}
                <span className="mw-place-num">{`#${index + 1} · ${place.heightM.toFixed(2)} m`}</span>
            </div>
            <dl className="mw-place-body">
                <dt>where</dt>
                <dd>{`x ${x.toFixed(2)}  y ${y.toFixed(2)}  z ${z.toFixed(2)}`}</dd>
                <dt>size</dt>
                <dd>{`${sx.toFixed(2)} × ${sy.toFixed(2)} × ${sz.toFixed(2)} m`}</dd>
                <dt>height</dt>
                <dd>{`${meters(place.heightM)} (${place.bottomZ.toFixed(2)} → ${place.topZ.toFixed(2)})`}</dd>
                <dt>seen</dt>
                <dd>{`${plural(place.observations.length, "time")} · ${seenAt(place.firstSeen, status)} → ${seenAt(place.lastSeen, status)}`}</dd>
            </dl>
        </button>
    )
}

export function Ask({ status }: { status: AppStatus }) {
    const [viewer, setViewer] = useState<Viewer | null>(null)
    const sceneRef = useRef<MemoryScene | null>(null)
    const [scene, setScene] = useState<MemoryScene | null>(null)
    const [text, setText] = useState("")
    // OWLv2's box score cutoff: lower finds more (and more wrong) things
    const [threshold, setThreshold] = useState(0.2)
    const [asking, setAsking] = useState<{ text: string; since: number } | null>(null)
    const [answer, setAnswer] = useState<Answer | null>(null)
    const [problem, setProblem] = useState<string | null>(null)
    const [selected, setSelected] = useState<number | null>(null)
    const [sceneState, setSceneState] = useState<"loading" | "ready" | string>("loading")
    const [now, setNow] = useState(Date.now())
    const shown = useRef("")
    const askingRef = useRef(false)
    askingRef.current = !!asking
    const generation = status.memory?.generation ?? 0

    // the scene, once the viewer is up
    useEffect(() => {
        if (!viewer) {
            return
        }
        const created = new MemoryScene(viewer)
        sceneRef.current = created
        setScene(created)
        return () => {
            created.dispose()
            sceneRef.current = null
            setScene(null)
        }
    }, [viewer])

    // the memory's cloud and path (again when it's regenerated)
    useEffect(() => {
        if (!scene) {
            return
        }
        let stale = false
        setSceneState("loading")
        Promise.all([api.cloud(), api.frames()])
            .then(([cloud, frames]) => {
                if (stale) {
                    return
                }
                scene.setCloud(cloud)
                scene.setPath(frames.frames)
                scene.frameAll()
                setSceneState("ready")
                if (shown.current) {
                    scene.focus(undefined, 0)
                }
            })
            .catch((error) => !stale && setSceneState(String(error.message ?? error)))
        return () => {
            stale = true
        }
    }, [scene, generation])

    const show = (next: Answer | null, fly = true) => {
        shown.current = answerKey(next)
        setAnswer(next)
        setSelected(null)
        const current = sceneRef.current
        if (current) {
            current.setPlaces(next?.places ?? [])
            if (fly && next?.places.length) {
                current.focus()
            }
        }
    }

    // the last answer (asked here before a refresh, or by Desktop's agent), and new ones the agent asks meanwhile
    useEffect(() => {
        if (!scene) {
            return
        }
        const poll = () =>
            api.answer()
                .then((last) => {
                    if (!askingRef.current && last && answerKey(last) !== shown.current) {
                        setText(last.query)
                        show(last)
                    }
                })
                .catch(() => {})
        poll()
        const timer = setInterval(poll, 2500)
        return () => clearInterval(timer)
    }, [scene])

    useEffect(() => {
        if (scene) {
            scene.onSelect = (id) => setSelected(id)
        }
    }, [scene])
    useEffect(() => {
        scene?.select(selected)
    }, [scene, selected])

    // the elapsed time while a question runs
    useEffect(() => {
        if (!asking) {
            return
        }
        const timer = setInterval(() => setNow(Date.now()), 100)
        return () => clearInterval(timer)
    }, [asking])

    const ask = async (question: string) => {
        const trimmed = question.trim()
        if (!trimmed || asking) {
            return
        }
        setText(trimmed)
        setProblem(null)
        setAsking({ text: trimmed, since: Date.now() })
        setNow(Date.now())
        try {
            show(await api.query(trimmed, threshold))
        } catch (error) {
            setProblem(String((error as Error).message ?? error))
        } finally {
            setAsking(null)
        }
    }
    const submit = (event: FormEvent) => {
        event.preventDefault()
        ask(text)
    }
    const focusPlace = (id: number) => {
        setSelected(id)
        scene?.focus([id], 900)
    }

    const places = answer?.places ?? []
    return (
        <div className="mw-ask">
            <div className="mw-scene">
                <Scene3D onViewer={setViewer} grid style={{ position: "absolute", inset: 0 }} />
                <div className="mw-query">
                    <form onSubmit={submit} data-query-form>
                        <span className="mw-query-icon">{asking ? <Spinner size={20} /> : <Icon name="search" size={20} />}</span>
                        <input
                            value={text}
                            onChange={(event) => setText(event.target.value)}
                            placeholder="Ask what's in the recording… e.g. traffic cone"
                            aria-label="Ask what's in the recording"
                            autoFocus
                            data-query
                        />
                        <label className="mw-threshold" title="Minimum box score (0–1): lower finds more, and more wrong, things">
                            min score
                            <NumberInput size="sm" value={threshold} onChange={setThreshold} min={0.01} max={1} step={0.05} precision={2} data-threshold />
                        </label>
                        <Button type="submit" variant="primary" disabled={!text.trim() || !!asking} data-action="ask">
                            {asking ? "Looking…" : "Find"}
                        </Button>
                    </form>
                    {!answer && !asking && (
                        <div className="mw-examples">
                            {EXAMPLES.map((example) => (
                                <Chip key={example} onClick={() => ask(example)}>{example}</Chip>
                            ))}
                        </div>
                    )}
                </div>
                {(sceneState !== "ready" || status.loading) && (
                    <Hud position="bottom">
                        {sceneState !== "ready" && sceneState !== "loading" ? (
                            <Alert tone="warn" title="Couldn't draw the memory">{sceneState}</Alert>
                        ) : (
                            <div className="mw-overlay-note">
                                <Spinner size={16} />
                                {status.loading ? "Loading the memory into RAM…" : "Drawing the memory…"}
                            </div>
                        )}
                    </Hud>
                )}
                <Hud position="bottom-left">
                    <Row gap={6}>
                        <IconButton icon="fit" variant="default" label="See the whole memory" onClick={() => scene?.frameAll()} />
                        {places.length > 0 && <IconButton icon="target" variant="default" label="Back to the answer" onClick={() => scene?.focus()} />}
                    </Row>
                </Hud>
            </div>
            <aside className="mw-side">
                <div className="mw-side-scroll">
                    {problem && <Alert tone="danger" title="That question failed" onClose={() => setProblem(null)}>{problem}</Alert>}
                    {asking && (
                        <Column gap={8} data-asking>
                            <Text>
                                Looking for <strong>“{asking.text}”</strong> in {status.loaded ? plural(status.loaded.frames, "keyframe") : "the memory"}…
                            </Text>
                            <Progress />
                            <Text small mono muted>{`${((now - asking.since) / 1000).toFixed(1)} s`}</Text>
                        </Column>
                    )}
                    {!asking && !answer && (
                        <EmptyState
                            icon="search"
                            title="Ask where something is"
                            message="Type anything the robot might have seen: “traffic cone”, “red chair”, “fire extinguisher”. Every match becomes a box in 3D, with the picture it was seen in."
                        />
                    )}
                    {!asking && answer && places.length === 0 && (
                        <EmptyState
                            icon="search"
                            title={`No “${answer.query}” found`}
                            message={answer.summary || "Nothing in the memory looked like it. Try other words (“cone” instead of “traffic cone”), or something you know the robot saw."}
                            data-empty-answer
                        >
                            <Text small mono muted>{`searched in ${(answer.timings.totalMs / 1000).toFixed(2)} s`}</Text>
                        </EmptyState>
                    )}
                    {!asking && answer && places.length > 0 && (
                        <>
                            <div className="mw-answer-head" data-answer-count={places.length}>
                                <span className="mw-count">{plural(places.length, "place")}</span>
                                <span className="mw-query-echo">{`for “${answer.query}”`}</span>
                            </div>
                            {answer.summary && <p className="mw-summary">{answer.summary}</p>}
                            <span className="mw-timings">
                                {`answered in ${(answer.timings.totalMs / 1000).toFixed(2)} s`}
                                {answer.unplaced > 0 ? ` · ${plural(answer.unplaced, "sighting")} couldn't be placed in 3D` : ""}
                            </span>
                            {places.map((place, index) => (
                                <PlaceCard key={place.id} place={place} index={index} query={answer.query} selected={selected === place.id} status={status} onClick={() => focusPlace(place.id)} />
                            ))}
                        </>
                    )}
                </div>
            </aside>
        </div>
    )
}
