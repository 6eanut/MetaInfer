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

const _BUILTINS = [
  ["text", TextField],
  ["textarea", TextAreaField],
  ["number", NumberField],
  ["select", SelectField],
  ["multiselect", MultiSelectField],
  ["radio", RadioField],
  ["file", FileField],
];

for (const [name, component] of _BUILTINS) {
  registerFormWidget(name, component);
  lockFormWidget(name);
}
