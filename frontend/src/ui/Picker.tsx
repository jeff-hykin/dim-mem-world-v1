// Pick a recording: Desktop's shared recordings (newest first), each with its memory's state from this app.
import { useEffect, useState } from "react"
import { Badge, Button, Card, EmptyState, IconButton, List, ListRow, Row, SearchInput, Spinner, Text } from "dimos-react"
import { getZenoh } from "dimos-react/dim-app/zenoh.js"
import { api, desktop, type AppRecording, type DesktopRecording, type MemoryStatus } from "../core/api.ts"
import { age, bytes, plural } from "../core/format.ts"
import { AppButton } from "./AppButton.tsx"

function MemoryBadge({ memory }: { memory: MemoryStatus | null | undefined }) {
    if (memory?.state === "ready") {
        return <Badge tone="ok" dot>{`memory · ${plural(memory.frames, "keyframe")}`}</Badge>
    }
    if (memory?.state === "partial") {
        return <Badge tone="warn" dot>{`memory ${memory.planned ? Math.round((memory.frames / memory.planned) * 100) : 0}% done`}</Badge>
    }
    return <Badge>no memory yet</Badge>
}

export function Picker({ current, onOpen, onBack }: { current: string | null; onOpen: (path: string) => Promise<void>; onBack: (() => void) | null }) {
    const [list, setList] = useState<DesktopRecording[] | null>(null)
    const [memories, setMemories] = useState<Map<string, MemoryStatus | null>>(new Map())
    const [dir, setDir] = useState("")
    const [problem, setProblem] = useState<string | null>(null)
    const [filter, setFilter] = useState("")
    const [opening, setOpening] = useState<string | null>(null)

    const load = () => {
        desktop
            .recordings()
            .then((body) => {
                setList([...body.recordings].sort((a, b) => b.modified - a.modified))
                setDir(body.dir)
                setProblem(null)
            })
            .catch((error) => setProblem(String(error.message ?? error)))
        api.recordings()
            .then((body) => {
                const map = new Map<string, MemoryStatus | null>()
                for (const recording of body.recordings as AppRecording[]) {
                    map.set(recording.path, recording.memory)
                    map.set(`id:${recording.id}`, recording.memory)
                }
                setMemories(map)
            })
            .catch(() => {})
    }
    useEffect(() => {
        load()
        // live: a recording added, renamed or deleted anywhere (the Controller, the Recordings app) shows up here
        try {
            const zenoh = getZenoh()
            const offs = [zenoh.subscribeDesktop("recordings", () => load()), zenoh.onReconnect(() => load())]
            return () => offs.forEach((off) => off())
        } catch {
            return undefined
        }
    }, [])

    const memoryOf = (recording: DesktopRecording) => memories.get(recording.path) ?? memories.get(`id:${recording.id}`) ?? null
    const shown = (list ?? []).filter((r) => r.name.toLowerCase().includes(filter.toLowerCase()))
    const open = async (recording: DesktopRecording) => {
        setOpening(recording.path)
        await onOpen(recording.path)
        setOpening(null)
    }

    return (
        <div className="mw-page mw-page-wide">
            <div className="mw-hero">
                <h1>Pick a recording</h1>
                <p className="mw-lead">
                    Memory World remembers what a robot saw. Pick a recording with a camera and a lidar; it gets a memory you can ask questions like
                    “where's the traffic cone?”, answered with boxes in 3D.
                </p>
            </div>
            {onBack && current && (
                <Card padded>
                    <Row justify="space-between" wrap>
                        <Text small>
                            Open now: <strong>{current}</strong>
                        </Text>
                        <Button size="sm" icon="arrow-left" onClick={onBack} data-action="back">
                            Back to it
                        </Button>
                    </Row>
                </Card>
            )}
            <Row>
                <SearchInput placeholder="filter recordings…" value={filter} onChange={setFilter} style={{ flex: 1 }} />
                <IconButton icon="refresh" size="sm" variant="default" label="Reload the list" onClick={load} />
            </Row>
            {dir && <Text small muted>{`Desktop's recordings folder: ${dir}`}</Text>}
            {problem && (
                <EmptyState
                    icon="warn"
                    tone="warn"
                    title="Couldn't list Desktop's recordings"
                    message={`${problem}. Desktop may be restarting; try again in a moment.`}
                    actions={<Button variant="primary" onClick={load}>Try again</Button>}
                    data-testid="recordings-error"
                />
            )}
            {!list && !problem && (
                <Row justify="center" style={{ padding: 32 }}>
                    <Spinner /> <Text muted>Reading Desktop's recordings…</Text>
                </Row>
            )}
            {list && !list.length && (
                <EmptyState
                    icon="record"
                    title="You don't have any recordings yet"
                    message="Record a robot with the Controller app (run a blueprint with a camera, open its Record tab, press Record), or add one with the Recordings app. New recordings show up here by themselves."
                    actions={
                        <>
                            <AppButton app="dim-controller" title="the Controller" variant="primary">Open the Controller</AppButton>
                            <AppButton app="dim-recordings" title="the Recordings app">Open Recordings</AppButton>
                        </>
                    }
                    data-testid="no-recordings"
                />
            )}
            {list && list.length > 0 && !shown.length && <Text small muted>No recordings match “{filter}”.</Text>}
            {shown.length > 0 && (
                <Card padded={false}>
                    <List>
                        {shown.map((recording) => (
                            <ListRow
                                key={recording.path}
                                icon={opening === recording.path ? <Spinner size={16} /> : "file"}
                                title={recording.name}
                                subtitle={`${recording.format} · ${bytes(recording.size)} · ${age(recording.modified)}`}
                                meta={<MemoryBadge memory={memoryOf(recording)} />}
                                selected={current === recording.name}
                                disabled={opening !== null}
                                onClick={() => open(recording)}
                                data-recording={recording.id}
                            />
                        ))}
                    </List>
                </Card>
            )}
        </div>
    )
}
