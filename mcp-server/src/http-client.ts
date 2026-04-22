import { request } from "undici";

const HOST = process.env.CONTEXTD_HTTP_HOST ?? "127.0.0.1";
const PORT = Number(process.env.CONTEXTD_HTTP_PORT ?? 8787);
const BASE = `http://${HOST}:${PORT}`;

export class HttpError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function parse(resp: {
  statusCode: number;
  body: { text(): Promise<string> };
}) {
  const text = await resp.body.text();
  if (resp.statusCode >= 400) {
    try {
      const j = JSON.parse(text) as {
        detail?: { code?: string; message?: string };
      };
      const d = j.detail ?? {};
      throw new HttpError(
        d.code ?? "INTERNAL",
        d.message ?? text,
        resp.statusCode,
      );
    } catch (e) {
      if (e instanceof HttpError) throw e;
      throw new HttpError("INTERNAL", text, resp.statusCode);
    }
  }
  return JSON.parse(text) as unknown;
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const resp = await request(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return (await parse(resp)) as T;
}

export async function get<T>(
  path: string,
  query?: Record<string, string | number | boolean>,
): Promise<T> {
  const url = new URL(`${BASE}${path}`);
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      url.searchParams.set(k, String(v));
    }
  }
  const resp = await request(url);
  return (await parse(resp)) as T;
}
