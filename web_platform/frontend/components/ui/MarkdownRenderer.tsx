"use client";

/**
 * Markdown Renderer Component
 *
 * Renders markdown content with syntax highlighting.
 */

import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import "highlight.js/styles/github-dark.css";
import { getImageUrl } from "../../lib/utils/image";

export interface MarkdownRendererProps {
  /** Markdown content to render */
  content: string;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Renders markdown content with GitHub Flavored Markdown and syntax highlighting.
 */
export function MarkdownRenderer({ content, className = "" }: MarkdownRendererProps) {
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          // Custom heading styles
          h1: (props) => (
            <h1 className="text-2xl font-bold mt-6 mb-4 text-white first:mt-0" {...props} />
          ),
          h2: (props) => (
            <h2 className="text-xl font-bold mt-5 mb-3 text-white first:mt-0" {...props} />
          ),
          h3: (props) => (
            <h3 className="text-lg font-semibold mt-4 mb-2 text-white first:mt-0" {...props} />
          ),
          h4: (props) => (
            <h4 className="text-base font-semibold mt-3 mb-2 text-white first:mt-0" {...props} />
          ),
          h5: (props) => (
            <h5 className="text-sm font-semibold mt-3 mb-2 text-zinc-200 first:mt-0" {...props} />
          ),
          h6: (props) => (
            <h6 className="text-sm font-medium mt-2 mb-1 text-zinc-300 first:mt-0" {...props} />
          ),

          // Paragraph styles
          p: (props) => (
            <p className="mb-3 text-zinc-200 leading-relaxed last:mb-0" {...props} />
          ),

          // List styles
          ul: (props) => (
            <ul className="list-disc list-outside mb-3 space-y-1.5 text-zinc-200 ml-5" {...props} />
          ),
          ol: (props) => (
            <ol className="list-decimal list-outside mb-3 space-y-1.5 text-zinc-200 ml-5" {...props} />
          ),
          li: (props) => <li className="leading-relaxed" {...props} />,

          // Code blocks
          code: ({
            inline,
            className: codeClassName,
            children,
            ...props
          }: {
            inline?: boolean;
            className?: string;
            children?: React.ReactNode;
          }) => {
            return inline ? (
              <code
                className="bg-zinc-700/50 text-zinc-200 px-1.5 py-0.5 rounded text-sm font-mono border border-zinc-600"
                {...props}
              >
                {children}
              </code>
            ) : (
              <code
                className={`${codeClassName || ""} block bg-zinc-900/50 p-4 rounded-lg overflow-x-auto text-sm font-mono border border-zinc-700 my-3`}
                {...props}
              >
                {children}
              </code>
            );
          },

          // Blockquote
          blockquote: (props) => (
            <blockquote className="border-l-4 border-emerald-500 pl-4 italic text-zinc-400 my-4" {...props} />
          ),

          // Links
          a: (props) => (
            <a
              className="text-blue-400 hover:text-blue-300 underline decoration-blue-500/50 hover:decoration-blue-400 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
              {...props}
            />
          ),

          // Strong (bold) text
          strong: (props) => <strong className="font-semibold text-white" {...props} />,

          // Emphasis (italic) text
          em: (props) => <em className="italic text-zinc-200" {...props} />,

          // Tables
          table: (props) => (
            <div className="overflow-x-auto my-4">
              <table className="min-w-full border-collapse border border-zinc-700" {...props} />
            </div>
          ),
          thead: (props) => <thead className="bg-zinc-800" {...props} />,
          th: (props) => (
            <th className="border border-zinc-700 px-4 py-2 text-left font-semibold text-white" {...props} />
          ),
          td: (props) => (
            <td className="border border-zinc-700 px-4 py-2 text-zinc-300" {...props} />
          ),

          // Horizontal rule
          hr: (props) => <hr className="my-6 border-t border-zinc-700" {...props} />,

          // Images (using img tag for dynamic markdown src)
          img: ({ src, alt, ...props }) => {
            // Normalize backend image paths to include API base URL
            // src can be string or Blob - only process if string
            const srcString = typeof src === "string" ? src : undefined;
            const resolvedSrc = srcString ? getImageUrl(srcString) ?? srcString : undefined;
            return (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                src={resolvedSrc}
                alt={alt || ""}
                className="max-w-full h-auto rounded-lg border border-zinc-700 my-4"
                loading="lazy"
                onError={(e) => {
                  // Handle broken images gracefully
                  const img = e.currentTarget;
                  img.style.display = "none";
                  const parent = img.parentElement;
                  if (parent && !parent.querySelector(".image-error")) {
                    const errorDiv = document.createElement("div");
                    errorDiv.className =
                      "image-error bg-red-900/20 border border-red-800 rounded-lg p-3 my-4";
                    errorDiv.innerHTML = `<p class="text-sm text-red-400">⚠️ Failed to load image: ${alt || src || "unknown"
                      }</p>`;
                    parent.appendChild(errorDiv);
                  }
                }}
                {...props}
              />
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
