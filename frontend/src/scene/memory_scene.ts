// The memory in 3D, drawn into dimos-react's Viewer (z-up, meters, the recording's world frame, so nothing is converted):
// the point cloud colored by its RGB, the camera's path, and an answer's places as bright boxes with numbered labels.
// After a question the camera flies to frame the places; clicking a box selects that place.
import * as THREE from "three"
import { FatLines, LabelPool, pushBox, type Viewer } from "dimos-react/three"
import type { Cloud, Keyframe, Place } from "../core/api.ts"

export const PLACE_COLOR = "#ff8a1f"
export const SELECTED_COLOR = "#22e4ff"
const PATH_COLOR = 0x7dd3fc

const cloudVertex = /* glsl */ `
uniform float uSize;
uniform float uPxPerMeter;
attribute vec3 color;
varying vec3 vColor;
void main() {
    vColor = color;
    vec4 view = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * view;
    gl_PointSize = clamp(uSize * uPxPerMeter / max(0.05, -view.z), 1.5, 9.0);
}
`
const cloudFragment = /* glsl */ `
varying vec3 vColor;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r = dot(p, p);
    if (r > 1.0) discard;
    // a hint of shading so dense areas read as surfaces
    gl_FragColor = vec4(vColor * (1.0 - 0.25 * r), 1.0);
}
`

const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
const easeOutBack = (t: number) => {
    const c = 1.70158
    return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2)
}

interface Flight {
    fromTarget: THREE.Vector3
    fromPosition: THREE.Vector3
    toTarget: THREE.Vector3
    toPosition: THREE.Vector3
    start: number
    ms: number
}

export class MemoryScene {
    readonly viewer: Viewer
    #root = new THREE.Group()
    #cloud: THREE.Points | null = null
    #cloudMaterial: THREE.ShaderMaterial
    #path: FatLines
    #edges: FatLines
    #glow: FatLines
    #fills = new THREE.Group()
    #labels = new LabelPool("dr-scene-label mw-place-label")
    #places: Place[] = []
    #selected: number | null = null
    #appearStart = 0
    #flight: Flight | null = null
    #frame = 0
    #disposed = false
    #raycaster = new THREE.Raycaster()
    #down: { x: number; y: number } | null = null
    onSelect: ((id: number) => void) | null = null

    constructor(viewer: Viewer) {
        this.viewer = viewer
        this.#cloudMaterial = new THREE.ShaderMaterial({
            vertexShader: cloudVertex,
            fragmentShader: cloudFragment,
            uniforms: { uSize: { value: 0.05 }, uPxPerMeter: viewer.pixelsPerMeter },
        })
        this.#path = new FatLines(viewer.resolution, { width: 2.5, color: PATH_COLOR, opacity: 0.8 })
        this.#path.object.renderOrder = 2
        // a wide translucent pass under the edges reads as a glow; both drawn over the cloud
        this.#glow = new FatLines(viewer.resolution, { width: 12, vertexColors: true, opacity: 0.28 })
        this.#edges = new FatLines(viewer.resolution, { width: 4.5, vertexColors: true })
        for (const lines of [this.#glow, this.#edges]) {
            lines.material.depthTest = false
            lines.material.transparent = true
            lines.object.renderOrder = 10
        }
        this.#root.add(this.#path.object, this.#fills, this.#glow.object, this.#edges.object, this.#labels.group)
        viewer.scene.add(this.#root)
        viewer.controls.addEventListener("start", this.#stopFlight)
        viewer.canvas.addEventListener("pointerdown", this.#onDown)
        viewer.canvas.addEventListener("pointerup", this.#onUp)
    }

    setCloud(cloud: Cloud | null) {
        if (this.#cloud) {
            this.#root.remove(this.#cloud)
            this.#cloud.geometry.dispose()
            this.#cloud = null
        }
        if (cloud && cloud.count) {
            const geometry = new THREE.BufferGeometry()
            geometry.setAttribute("position", new THREE.BufferAttribute(cloud.positions, 3))
            geometry.setAttribute("color", new THREE.BufferAttribute(cloud.colors, 3, true))
            geometry.computeBoundingSphere()
            this.#cloud = new THREE.Points(geometry, this.#cloudMaterial)
            this.#cloud.frustumCulled = false
            this.#root.add(this.#cloud)
        }
        this.viewer.requestRender()
    }

    setPath(frames: Keyframe[]) {
        this.#path.clear()
        for (let index = 1; index < frames.length; index++) {
            const [a, b] = [frames[index - 1].position, frames[index].position]
            this.#path.push(a[0], a[1], a[2], b[0], b[1], b[2])
        }
        this.#path.commit()
        this.viewer.requestRender()
    }

    /** the answer's places; they pop in one after another */
    setPlaces(places: Place[]) {
        this.#places = places
        this.#selected = null
        for (const child of [...this.#fills.children]) {
            const mesh = child as THREE.Mesh
            mesh.geometry.dispose()
            ;(mesh.material as THREE.Material).dispose()
            this.#fills.remove(mesh)
        }
        for (const place of places) {
            const mesh = new THREE.Mesh(
                new THREE.BoxGeometry(1, 1, 1),
                new THREE.MeshBasicMaterial({ color: PLACE_COLOR, transparent: true, opacity: 0.16, depthWrite: false }),
            )
            mesh.position.set(...place.center)
            mesh.userData = { noPick: true, place: place.id }
            mesh.renderOrder = 9
            this.#fills.add(mesh)
        }
        this.#appearStart = performance.now()
        this.#animate()
    }

    select(id: number | null) {
        this.#selected = id
        this.#draw(performance.now())
        this.viewer.requestRender()
    }

    /** flies to frame these places (all of them when none are given) */
    focus(ids?: number[], ms = 1100) {
        const places = ids ? this.#places.filter((place) => ids.includes(place.id)) : this.#places
        if (!places.length) {
            return
        }
        const box = new THREE.Box3()
        for (const place of places) {
            box.union(
                new THREE.Box3().setFromCenterAndSize(
                    new THREE.Vector3(...place.center),
                    new THREE.Vector3(...place.size.map((s) => Math.max(0.3, s))),
                ),
            )
        }
        const center = box.getCenter(new THREE.Vector3())
        const radius = Math.max(0.6, box.getSize(new THREE.Vector3()).length() / 2)
        const camera = this.viewer.camera
        const halfFov = THREE.MathUtils.degToRad(camera.fov) / 2
        const fit = Math.min(1, camera.aspect)
        const distance = THREE.MathUtils.clamp((radius / Math.sin(halfFov) / fit) * 1.7, 4, 400)
        // keep the heading, but look down at a pleasant 35-55°
        const direction = camera.position.clone().sub(this.viewer.controls.target)
        const flat = new THREE.Vector2(direction.x, direction.y)
        if (flat.lengthSq() < 1e-6) {
            flat.set(-1, -1)
        }
        flat.normalize()
        const elevation = THREE.MathUtils.clamp(Math.atan2(direction.z, Math.hypot(direction.x, direction.y)), 0.6, 0.95)
        const to = new THREE.Vector3(flat.x * Math.cos(elevation), flat.y * Math.cos(elevation), Math.sin(elevation))
        this.flyTo(center, center.clone().addScaledVector(to, distance), ms)
    }

    /** the whole memory, from the default 3/4 view */
    frameAll() {
        this.#stopFlight()
        this.viewer.frameAll()
    }

    flyTo(target: THREE.Vector3, position: THREE.Vector3, ms = 1100) {
        const reduced = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches
        this.#flight = {
            fromTarget: this.viewer.controls.target.clone(),
            fromPosition: this.viewer.camera.position.clone(),
            toTarget: target,
            toPosition: position,
            start: performance.now(),
            ms: reduced ? 1 : Math.max(1, ms),
        }
        this.#animate()
    }

    #stopFlight = () => {
        this.#flight = null
    }

    #animate() {
        if (!this.#frame && !this.#disposed) {
            this.#frame = requestAnimationFrame(this.#step)
        }
    }

    #step = (now: number) => {
        this.#frame = 0
        let busy = false
        const flight = this.#flight
        if (flight) {
            const t = Math.min(1, (now - flight.start) / flight.ms)
            const e = easeInOut(t)
            const controls = this.viewer.controls
            controls.target.lerpVectors(flight.fromTarget, flight.toTarget, e)
            this.viewer.camera.position.lerpVectors(flight.fromPosition, flight.toPosition, e)
            this.viewer.camera.lookAt(controls.target)
            controls.update()
            if (t >= 1) {
                this.#flight = null
            } else {
                busy = true
            }
        }
        busy = this.#draw(now) || busy
        this.viewer.requestRender()
        if (busy) {
            this.#animate()
        }
    }

    /** redraws the boxes; true while some are still popping in */
    #draw(now: number): boolean {
        this.#edges.clear()
        this.#glow.clear()
        this.#labels.begin()
        let busy = false
        const matrix = new THREE.Matrix4()
        const orange = new THREE.Color(PLACE_COLOR)
        const cyan = new THREE.Color(SELECTED_COLOR)
        this.#places.forEach((place, index) => {
            const t = THREE.MathUtils.clamp((now - this.#appearStart - index * 140) / 480, 0, 1)
            if (t < 1) {
                busy = true
            }
            const grow = Math.max(0.001, easeOutBack(t))
            const selected = place.id === this.#selected
            const color = selected ? cyan : orange
            const size = new THREE.Vector3(...place.size.map((s) => Math.max(0.08, s) * grow))
            matrix.makeTranslation(place.center[0], place.center[1], place.center[2])
            if (t > 0) {
                pushBox(this.#glow, matrix, size, color)
                pushBox(this.#edges, matrix, size, color)
            }
            const fill = this.#fills.children[index] as THREE.Mesh | undefined
            if (fill) {
                fill.scale.copy(size)
                fill.visible = t > 0
                const material = fill.material as THREE.MeshBasicMaterial
                material.color.copy(color)
                material.opacity = selected ? 0.28 : 0.16
            }
            if (t > 0.6) {
                const top = new THREE.Vector3(place.center[0], place.center[1], place.center[2] + size.z / 2 + 0.05)
                const label = this.#labels.place(`#${index + 1} · ${place.heightM.toFixed(2)} m`, top, selected ? SELECTED_COLOR : PLACE_COLOR)
                label.element.classList.toggle("mw-selected", selected)
            }
        })
        this.#edges.commit()
        this.#glow.commit()
        this.#labels.end()
        return busy
    }

    #onDown = (event: PointerEvent) => {
        this.#down = { x: event.clientX, y: event.clientY }
    }

    #onUp = (event: PointerEvent) => {
        const down = this.#down
        this.#down = null
        if (!down || Math.hypot(event.clientX - down.x, event.clientY - down.y) > 4 || !this.onSelect) {
            return
        }
        const rect = this.viewer.canvas.getBoundingClientRect()
        const ndc = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1)
        this.#raycaster.setFromCamera(ndc, this.viewer.camera)
        const hit = this.#raycaster.intersectObjects(this.#fills.children, false)[0]
        if (hit) {
            this.onSelect(hit.object.userData.place as number)
        }
    }

    dispose() {
        this.#disposed = true
        cancelAnimationFrame(this.#frame)
        this.viewer.controls.removeEventListener("start", this.#stopFlight)
        this.viewer.canvas.removeEventListener("pointerdown", this.#onDown)
        this.viewer.canvas.removeEventListener("pointerup", this.#onUp)
        this.setCloud(null)
        this.setPlaces([])
        this.#labels.dispose()
        this.viewer.scene.remove(this.#root)
    }
}
