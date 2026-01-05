import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    ignores: [
      "node_modules/**",
      ".next/**",
      "out/**",
      "build/**",
      "next-env.d.ts",
      "lib/types/openapi.d.ts", // Generated file
    ],
  },
  {
    rules: {
      // Allow 'any' in transformer/API client code where we're bridging between
      // OpenAPI-generated types and our UI types
      "@typescript-eslint/no-explicit-any": ["warn", {
        "ignoreRestArgs": true,
        "fixToUnknown": false
      }],
      // Allow unescaped quotes in JSX
      "react/no-unescaped-entities": "off",
      // Allow img tags (we use them for dynamic backend-served images)
      "@next/next/no-img-element": "warn",
    },
  },
];

export default eslintConfig;
