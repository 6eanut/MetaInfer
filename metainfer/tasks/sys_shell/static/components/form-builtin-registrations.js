// Registers the shell's built-in form widgets with the form-registry,
// then locks them so plugins can't accidentally hijack primitives
// like "text" or "number".
//
// This module is imported eagerly from main.js BEFORE any view renders,
// guaranteeing the built-ins are available when FormRenderer first runs.
//
// Plugin extensions (form-overrides.js) load AFTER this module via
// form-overrides-loader.js.

import { registerFormWidget, lockFormWidget } from "app/form-registry";
import { TextField } from "app/form-fields/text-field";
import { TextAreaField } from "app/form-fields/textarea-field";
import { NumberField } from "app/form-fields/number-field";
import { SelectField } from "app/form-fields/select-field";
import { MultiSelectField } from "app/form-fields/multiselect-field";
import { RadioField } from "app/form-fields/radio-field";
import { FileField } from "app/form-fields/file-field";
import { WorkerMultiSelect } from "app/worker-multiselect";

// Primitives: locked so plugins cannot hijack text/number/etc.
const _PRIMITIVES = [
  ["text", TextField],
  ["textarea", TextAreaField],
  ["number", NumberField],
  ["select", SelectField],
  ["multiselect", MultiSelectField],
  ["radio", RadioField],
  ["file", FileField],
];

for (const [name, component] of _PRIMITIVES) {
  registerFormWidget(name, component);
  lockFormWidget(name);
}

// Shell-level composite widgets (not locked — plugins may override if needed).
// worker-multiselect dynamically loads from /api/cluster/workers.
registerFormWidget("worker-multiselect", WorkerMultiSelect);
