// The Memory World page. All state lives server-side (api/status), polled: fast while something runs (a model download,
// a generation, the memory loading), slowly otherwise so what Desktop's agent does shows up here too. The screen follows
// the state: no recording → pick one; not compatible / no memory → the setup steps; a memory → ask it questions.
import { useCallback, useEffect, useRef, useState } from "react"
import { AppShell, Badge, Button, EmptyState, Row, Spinner, Text, TopBar, toast, useConfirm } from "dimos-react"
import { api, type AppStatus } from "./core/api.ts"
import { plural } from "./core/format.ts"
import { Picker } from "./ui/Picker.tsx"
import { isGenerating, Setup } from "./ui/Setup.tsx"
import { Ask } from "./ui/Ask.tsx"

const FAST_MS = 500
const SLOW_MS = 3000

const busy = (status: AppStatus | null) => !!status && (status.models.downloading || isGenerating(status) || status.loading)

export function App() {
    const [status, setStatus] = useState<AppStatus | null>(null)
    const [unreachable, setUnreachable] = useState<string | null>(null)
    const [picking, setPicking] = useState(false)
    const [forgetting, setForgetting] = useState(false)
    const confirm = useConfirm()
    const statusRef = useRef<AppStatus | null>(null)
    statusRef.current = status

    const refresh = useCallback(async () => {
        try {
            const next = await api.status()
            setStatus(next)
            setUnreachable(null)
            return next
        } catch (error) {
            setUnreachable(String((error as Error).message ?? error))
            return null
        }
    }, [])

    useEffect(() => {
        let timer = 0
        let stopped = false
        const tick = async () => {
            await refresh()
            if (!stopped) {
                timer = window.setTimeout(tick, busy(statusRef.current) ? FAST_MS : SLOW_MS)
            }
        }
        tick()
        return () => {
            stopped = true
            clearTimeout(timer)
        }
    }, [refresh])

    // a status pushed by an action (generate, cancel, open) speeds the polling up at once
    const onStatus = useCallback((next: AppStatus) => setStatus(next), [])

    const open = async (path: string) => {
        try {
            setStatus(await api.open(path))
            setPicking(false)
        } catch (error) {
            toast({ message: `Couldn't open it: ${(error as Error).message}`, tone: "danger", durationMs: 7000 })
        }
    }

    const forget = async () => {
        const ok = await confirm({
            title: "Forget this recording's memory?",
            message: "Its keyframes and embeddings are removed from the recording. The recording itself stays. You can generate the memory again any time.",
            danger: true,
            confirmLabel: "Forget memory",
        })
        if (!ok) {
            return
        }
        setForgetting(true)
        try {
            setStatus(await api.forget())
            toast({ message: "Memory forgotten", tone: "info" })
        } catch (error) {
            toast({ message: `Couldn't forget it: ${(error as Error).message}`, tone: "danger", durationMs: 7000 })
        } finally {
            setForgetting(false)
        }
    }

    const recording = status?.recording ?? null
    const memory = status?.memory ?? null
    const ready = memory?.state === "ready"
    const showPicker = !!status && (!recording || picking)

    const right = recording && !showPicker && (
        <Row gap={8}>
            <Button size="sm" icon="folder" onClick={() => setPicking(true)} disabled={isGenerating(status)} title={isGenerating(status) ? "Cancel the generation first" : undefined} data-action="change-recording">
                Change recording
            </Button>
            {memory && memory.state !== "none" && (
                <Button size="sm" variant="ghost" icon="trash" loading={forgetting} disabled={isGenerating(status)} onClick={forget} data-action="forget">
                    Forget memory
                </Button>
            )}
        </Row>
    )
    const left = recording && !showPicker && (
        <Row gap={8}>
            <Text muted>/</Text>
            <Text as="strong" data-recording-name>{recording.name}</Text>
            {ready && <Badge tone="ok" dot>{`memory · ${plural(memory.frames, "keyframe")}`}</Badge>}
            {memory?.state === "partial" && <Badge tone="warn" dot>memory partly done</Badge>}
            {isGenerating(status) && <Badge tone="info" dot>generating…</Badge>}
        </Row>
    )

    let body
    if (!status) {
        body = unreachable ? (
            <EmptyState
                layer
                icon="warn"
                tone="warn"
                title="Can't reach Memory World's server"
                message={`${unreachable}. It may still be starting; this page retries by itself.`}
                actions={<Button variant="primary" onClick={refresh}>Try again</Button>}
            />
        ) : (
            <div className="mw-center">
                <Row gap={10}>
                    <Spinner /> <Text muted>Starting Memory World…</Text>
                </Row>
            </div>
        )
    } else if (showPicker) {
        body = <Picker current={recording?.name ?? null} onOpen={open} onBack={recording ? () => setPicking(false) : null} />
    } else if (ready) {
        body = <Ask status={status} />
    } else {
        body = <Setup status={status} onStatus={onStatus} onPickAnother={() => setPicking(true)} />
    }

    return (
        <AppShell insetBottom topBar={<TopBar title="Memory World" icon="cube" left={left} right={right} />}>
            {unreachable && status && (
                <div style={{ position: "absolute", top: 8, right: 8, zIndex: 50 }}>
                    <Badge tone="warn" dot>{`server unreachable · retrying`}</Badge>
                </div>
            )}
            {body}
        </AppShell>
    )
}
