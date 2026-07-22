// Discovers and loads every plugin's ``form-overrides.js`` module
// before the SPA renders.
//
// Convention: a plugin that ships custom form widgets adds an
// ``importmap_entries`` key of the form
// ``"app/form-overrides/<plugin-type>" -> <url>`` in its server-side
// WebPlugin descriptor. At runtime the merged importmap is inlined
// into ``index.html`` as ``<script type="importmap" id="metainfer-importmap">``.
//
// This loader reads that block, filters for the ``app/form-overrides/``
// prefix, and dynamically imports each module. Modules are expected to
// call ``registerFormWidget(name, component)`` for each widget they
// want to register or override.
//
// Last-wins: if two plugins register the same name, the one whose
// module resolves later wins. This mirrors normal ES module evaluation
// order — deterministic given the importmap.

const _PREFIX = "app/form-overrides/";

function _readImportmap() {
  const script = document.getElementById("metainfer-importmap");
  if (!script) return {};
  try {
    const parsed = JSON.parse(script.textContent || "{}");
    return parsed.imports || {};
  } catch (_) {
    return {};
  }
}

export function listPluginOverrideModules() {
  const imports = _readImportmap();
  return Object.keys(imports)
    .filter((k) => k.startsWith(_PREFIX) && k.length > _PREFIX.length)
    .sort();
}

export async function loadAllPluginOverrides() {
  const keys = listPluginOverrideModules();
  if (keys.length === 0) return;
  // allSettled: a broken plugin module must not prevent the SPA from
  // rendering. The browser console surfaces the failure for debugging.
  await Promise.allSettled(keys.map((k) => import(k)));
}
