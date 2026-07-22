// port-model form widget registrations.
//
// Discovered and imported by the shell's form-overrides-loader (see
// app/form-overrides-loader). Each plugin that wants custom form
// widgets ships a module like this one and registers an
// ``importmap_entries`` key ``"app/form-overrides/<type>"`` pointing
// at it from its WebPlugin descriptor.
//
// Adding a new widget:
//   1. Put the component file under ``static/components/<name>.js``.
//   2. Import it here and call ``registerFormWidget("<name>", Cmp)``.
//   3. Reference it from ``form.yaml`` via ``override_component: <name>``.

import { registerFormWidget } from "app/form-registry";
import { KvListPathNotes } from "./components/kv-list-path-notes.js";

registerFormWidget("kv-list-path-notes", KvListPathNotes);
