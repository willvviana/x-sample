import { useState, type SyntheticEvent } from "react";

interface Props {
  notice: string | undefined;
  onConnect: (token: string) => void;
}

export function ConnectScreen({ notice, onConnect }: Props) {
  const [value, setValue] = useState("");

  const submit = (event: SyntheticEvent) => {
    event.preventDefault();
    const token = value.trim();
    if (token) onConnect(token);
  };

  return (
    <main className="connect" data-testid="connect-screen">
      <h1>X-Sample</h1>
      <p>
        Open X-Sample with the link printed by <code>xsample serve</code> — it contains a one-time
        access token for this engine session.
      </p>
      {notice && (
        <p className="notice" role="alert">
          {notice}
        </p>
      )}
      <form onSubmit={submit}>
        <label htmlFor="token">Or paste the token</label>
        <div className="row">
          <input
            id="token"
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
            }}
          />
          <button type="submit" disabled={!value.trim()}>
            Connect
          </button>
        </div>
      </form>
    </main>
  );
}
