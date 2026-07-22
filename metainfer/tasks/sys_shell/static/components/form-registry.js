// Form widget registry — the single dispatch table shared by the shell
// and every task plugin's form widgets.
//
// Lifecycle / ordering:
//
//   1. ``form-builtin-registrations.js`` imports each built-in field
//      component and registers it here under a stable name, then locks
//      the name. Locking prevents accidental overrides of the shell's
//      primitives (``text``, ``textarea``, ...).
//   2. Each plugin that ships custom widgets provides a
//      ``form-overrides.js`` module. ``form-overrides-loader.js``
//      discovers those modules via the importmap and imports them
//      before the SPA renders. Each module calls
//      ``registerFormWidget(name, component)`` for every widget it
//      wants to add or override. Last-wins semantics: a plugin that
//      registers the same name as another plugin wins if it loads
//      later (registration order is the order importmap resolves).
//
// Resolution: ``getWidgetForField(field)`` returns the component to use
// for a given schema field. ``override_component`` (when present) takes
// precedence over ``field.type``. This lets a plugin swap the widget
// for a specific field without rewriting the renderer.

const _components = new Map(); // name -> { component, locked }
const _order = []; // names in registration order, for diagnostics

export function registerFormWidget(name, component, opts = {}) {
  if (typeof name !== "string" || !name) {
    throw new Error("registerFormWidget: name must be a non-empty string");
  }
  if (typeof component !== "function") {
    throw new Error(`registerFormWidget(${name}): component must be a function`);
  }
  const existing = _components.get(name);
  if (existing && existing.locked && !opts.force) {
    // Locked built-ins cannot be silently overwritten — protects the
    // shell's primitives from accidental hijack. A plugin that
    // genuinely needs to replace a built-in must pass ``{force: true}``
    // and accept the explicit signal.
    console.warn(
      `[form-registry] refusing to override locked widget "${name}"; ` +
      `pass {force: true} to override.`
    );
    return false;
  }
  _components.set(name, { component, locked: !!opts.locked });
  if (!_order.includes(name)) _order.push(name);
  return true;
}

export function lockFormWidget(name) {
  const entry = _components.get(name);
  if (!entry) {
    console.warn(`[form-registry] cannot lock unknown widget "${name}"`);
    return false;
  }
  entry.locked = true;
  return true;
}

export function getFormWidget(name) {
  return _components.get(name)?.component ?? null;
}

export function hasFormWidget(name) {
  return _components.has(name);
}

export function registeredWidgetNames() {
  return [..._order];
}

// Resolve which widget to render for a schema field. Returns a Preact
// component function, or null when nothing matches (the renderer
// surfaces a friendly fallback in that case).
//
// Precedence: explicit ``override_component`` first (lets a plugin
// swap the widget per-field without renaming the type), then
// ``field.type`` (the default widget family).
export function getWidgetForField(field) {
  if (!field) return null;
  if (field.override_component) {
    const c = getFormWidget(field.override_component);
    if (c) return c;
  }
  if (field.type) {
    const c = getFormWidget(field.type);
    if (c) return c;
  }
  return null;
}
