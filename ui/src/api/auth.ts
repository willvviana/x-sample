// The engine token arrives in the launch URL fragment (#token=...), which browsers never
// send to servers. It is moved to sessionStorage and removed from the address bar.

const STORAGE_KEY = "xsample.token";

function readStored(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function storeToken(token: string): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, token);
  } catch {
    // Storage blocked: the token lives only in memory for this page load.
  }
}

export function clearToken(): void {
  try {
    sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}

export function captureToken(): string | null {
  const params = new URLSearchParams(window.location.hash.slice(1));
  const fromHash = params.get("token");
  if (fromHash) {
    storeToken(fromHash);
    window.history.replaceState(null, "", window.location.pathname + window.location.search);
    return fromHash;
  }
  return readStored();
}
