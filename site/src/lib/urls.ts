const base = import.meta.env.BASE_URL;

function normalizedRelativePath(path: string): string {
  const segments = path.split("/");
  if (
    path === "" ||
    path.startsWith("/") ||
    path.includes("\\") ||
    segments.some((segment) => segment === "" || segment === "." || segment === "..")
  ) {
    throw new Error(`invalid site-relative path: ${path}`);
  }
  return path;
}

export function routeHref(route = ""): string {
  const path = route.replace(/^\/+|\/+$/g, "");
  if (path !== "" && !/^[a-z0-9_-]+(?:\/[a-z0-9_-]+)*$/.test(path)) {
    throw new Error(`invalid site route: ${route}`);
  }
  return path === "" ? base : `${base}${path}/`;
}

/** A file of the verified release evidence, copied unchanged to artifacts/. */
export function artifactHref(path: string): string {
  return `${base}artifacts/${normalizedRelativePath(path)}`;
}

export function siteFileHref(path: string): string {
  return `${base}${normalizedRelativePath(path)}`;
}
