/**
 * ApiSecretGate Component
 *
 * Guards the entire application with API secret validation.
 * Users must enter the correct API secret before accessing any features.
 * This is the FIRST layer of security (before user login).
 */

"use client";

import { useState, useEffect } from "react";
import { API_CONFIG, API_SECRET_CONFIG } from "@/lib/config/api";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Card } from "../ui/Card";
import { Lock, AlertCircle, CheckCircle } from "lucide-react";

interface ApiSecretGateProps {
  children: React.ReactNode;
}

export function ApiSecretGate({ children }: ApiSecretGateProps) {
  const [secret, setSecret] = useState("");
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState("");
  const [isValidated, setIsValidated] = useState(false);
  const [isChecking, setIsChecking] = useState(true);

  // Check if secret is already stored
  useEffect(() => {
    const storedSecret = API_SECRET_CONFIG.getSecret();
    if (storedSecret) {
      // Validate the stored secret
      validateStoredSecret(storedSecret);
    } else {
      setIsChecking(false);
    }
  }, []);

  const validateStoredSecret = async (storedSecret: string) => {
    try {
      const response = await fetch(
        `${API_CONFIG.baseURL}/api/system/validate-secret`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Secret": storedSecret, // Send in header, not body!
          },
        }
      );

      const data = await response.json();

      if (data.valid) {
        setIsValidated(true);
      } else {
        // Stored secret is invalid, clear it
        API_SECRET_CONFIG.clearSecret();
      }
    } catch (err) {
      console.error("Failed to validate stored secret:", err);
      // On error, assume invalid and clear
      API_SECRET_CONFIG.clearSecret();
    } finally {
      setIsChecking(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsValidating(true);

    try {
      const response = await fetch(
        `${API_CONFIG.baseURL}/api/system/validate-secret`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Secret": secret, // Send in header, not body!
          },
        }
      );

      const data = await response.json();

      if (data.valid) {
        // Store the valid secret
        API_SECRET_CONFIG.setSecret(secret);
        setIsValidated(true);
      } else {
        setError("Invalid API secret. Please try again.");
      }
    } catch (err) {
      setError("Failed to validate secret. Please check your connection.");
      console.error("Validation error:", err);
    } finally {
      setIsValidating(false);
    }
  };

  const handleClearSecret = () => {
    API_SECRET_CONFIG.clearSecret();
    setIsValidated(false);
    setSecret("");
    setError("");
  };

  // Show loading while checking stored secret
  if (isChecking) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="text-white">
          <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p className="text-zinc-400">Checking credentials...</p>
        </div>
      </div>
    );
  }

  // If validated, render the app
  if (isValidated) {
    return <>{children}</>;
  }

  // Show API secret entry form
  return (
    <div className="min-h-screen bg-zinc-950 flex items-center justify-center p-4">
      <Card className="w-full max-w-md p-8 bg-zinc-900 border-zinc-800">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-500/10 rounded-full mb-4">
            <Lock className="h-8 w-8 text-blue-500" />
          </div>
          <h1 className="text-2xl font-bold text-white mb-2">API Access Required</h1>
          <p className="text-zinc-400 text-sm">
            Enter the API secret key to access MedRAX platform
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label
              htmlFor="api-secret"
              className="block text-sm font-medium text-zinc-300 mb-2"
            >
              API Secret Key
            </label>
            <Input
              id="api-secret"
              type="password"
              value={secret}
              onChange={(e) => setSecret(e.target.value)}
              placeholder="Enter your API secret..."
              className="font-mono"
              autoFocus
              required
            />
            <p className="mt-2 text-xs text-zinc-500">
              Please contact the administrator to get the API secret key.
            </p>
          </div>

          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0" />
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          <Button
            type="submit"
            variant="primary"
            className="w-full"
            disabled={isValidating || !secret}
          >
            {isValidating ? "Validating..." : "Access Platform"}
          </Button>
        </form>

        <div className="mt-8 p-4 bg-blue-500/5 border border-blue-500/10 rounded-lg">
          <div className="flex items-start gap-2">
            <CheckCircle className="h-5 w-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="text-xs text-zinc-400">
              <p className="font-semibold text-blue-400 mb-1">Security Note:</p>
              <p>
                The API secret is stored locally in your browser and sent with
                every request to verify access to the backend API.
              </p>
            </div>
          </div>
        </div>

        {/* Developer helper */}
        <div className="mt-4 text-center">
          <button
            type="button"
            onClick={handleClearSecret}
            className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            Clear stored secret
          </button>
        </div>
      </Card>
    </div>
  );
}
