// A button that opens another Desktop app; when that app isn't installed it offers the App Store instead, and outside
// Desktop it isn't shown.
import { useEffect, useState } from "react"
import { Button } from "dimos-react"
import { appInstalled, openApp, underDesktop } from "dimos-react/dim-app/desktop.js"

export function AppButton({ app, title, children, variant }: { app: string; title: string; children: string; variant?: "primary" | "default" }) {
    const [installed, setInstalled] = useState(true)
    useEffect(() => {
        appInstalled(app).then(setInstalled, () => setInstalled(false))
    }, [app])
    if (!underDesktop()) {
        return null
    }
    return installed
        ? <Button variant={variant} onClick={() => openApp(app)} data-app={app}>{children}</Button>
        : <Button variant={variant} onClick={() => openApp("appstore")} data-app="appstore">{`Install ${title} from the App Store`}</Button>
}
