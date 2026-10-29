import { useCallback, useEffect, useMemo, useState } from "react";

import { captureToken, clearToken, storeToken } from "./api/auth";
import { EngineClient } from "./api/client";
import { ConnectScreen } from "./components/ConnectScreen";
import { Dashboard } from "./components/Dashboard";

export function App() {
  const [token, setToken] = useState<string | null>(() => captureToken());
  const [notice, setNotice] = useState<string>();
  const client = useMemo(() => (token ? new EngineClient(token) : null), [token]);

  // Opening the launch link in an already-open tab only changes the fragment.
  useEffect(() => {
    const onHashChange = () => {
      const captured = captureToken();
      if (captured) {
        setNotice(undefined);
        setToken(captured);
      }
    };
    window.addEventListener("hashchange", onHashChange);
    return () => {
      window.removeEventListener("hashchange", onHashChange);
    };
  }, []);

  const handleUnauthorized = useCallback(() => {
    clearToken();
    setToken(null);
    setNotice("The engine rejected the token. It changes every time the engine starts.");
  }, []);

  const handleConnect = useCallback((value: string) => {
    storeToken(value);
    setNotice(undefined);
    setToken(value);
  }, []);

  if (!client) return <ConnectScreen notice={notice} onConnect={handleConnect} />;
  return <Dashboard client={client} onUnauthorized={handleUnauthorized} />;
}
