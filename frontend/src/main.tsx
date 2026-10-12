import { createRoot } from "react-dom/client"
import { AppProvider } from "dimos-react"
import { captureErrors } from "./core/errors.js"
import "./styles.css"
import { App } from "./App.tsx"

// uncaught errors reach Desktop's error feed, where its agent sees them
captureErrors()

createRoot(document.getElementById("root")!).render(
    <AppProvider toastPosition="bottom">
        <App />
    </AppProvider>,
)
