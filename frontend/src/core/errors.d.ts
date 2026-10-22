// Types for the vendored dim-app errors.js (v0.23.1).
export declare function reportError(
    message: string,
    detail?: unknown,
    options?: { level?: "error" | "warning"; source?: string },
): Promise<{ id: number; count: number } | null>
export declare function captureErrors(options?: { source?: string }): void
