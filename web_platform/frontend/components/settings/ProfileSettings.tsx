/**
 * ProfileSettings Component
 *
 * Allows doctor to update:
 * - Name
 * - Password
 */

"use client";

import { useState, useEffect } from "react";
import { useAuthStore } from "../../lib/store/authStore";
import { updateDoctor, updatePassword } from "../../lib/api/doctors";
import { Input } from "../ui/Input";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { validation } from "../../lib/utils/validation";

export function ProfileSettings() {
  const { doctor, token, setAuth } = useAuthStore();
  const isPasswordUpdateSupported = false;

  // Name Update State
  const [name, setName] = useState(doctor?.name || "");
  const [nameError, setNameError] = useState<string | null>(null);
  const [isUpdatingName, setIsUpdatingName] = useState(false);
  const [nameSuccess, setNameSuccess] = useState(false);

  useEffect(() => {
    setName(doctor?.name || "");
  }, [doctor]);

  // Password Update State
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [isUpdatingPassword, setIsUpdatingPassword] = useState(false);
  const [passwordSuccess, setPasswordSuccess] = useState(false);

  const handleNameUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!doctor) return;

    setNameError(null);
    setNameSuccess(false);

    const error = validation.doctorName(name);
    if (error) {
      setNameError(error);
      return;
    }

    setIsUpdatingName(true);
    try {
      const updatedDoctor = await updateDoctor({ name });
      const activeToken = token || localStorage.getItem("medrax_auth_token") || "";
      if (activeToken) {
        setAuth(updatedDoctor, activeToken);
      }
      setNameSuccess(true);
      setTimeout(() => setNameSuccess(false), 3000);
    } catch (err) {
      setNameError(err instanceof Error ? err.message : "Failed to update name");
    } finally {
      setIsUpdatingName(false);
    }
  };

  const handlePasswordUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!doctor) return;

    setPasswordError(null);
    setPasswordSuccess(false);

    if (!isPasswordUpdateSupported) {
      setPasswordError("Password updates are not available yet.");
      return;
    }

    // Validate
    const currentError = validation.password(currentPassword);
    if (currentError) {
      setPasswordError(currentError);
      return;
    }

    const newError = validation.password(newPassword);
    if (newError) {
      setPasswordError(`New password: ${newError}`);
      return;
    }

    if (newPassword !== confirmPassword) {
      setPasswordError("New passwords do not match");
      return;
    }

    setIsUpdatingPassword(true);
    try {
      await updatePassword({ currentPassword, newPassword });
      setPasswordSuccess(true);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setTimeout(() => setPasswordSuccess(false), 3000);
    } catch (err) {
      setPasswordError(err instanceof Error ? err.message : "Failed to update password");
    } finally {
      setIsUpdatingPassword(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">Profile Settings</h2>
        <p className="text-sm text-zinc-400">Update your personal information and security settings</p>
      </div>

      {/* Name Update */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Update Name</h3>
        <form onSubmit={handleNameUpdate} className="space-y-4">
          <Input
            label="Name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            error={nameError || undefined}
            disabled={isUpdatingName}
          />

          {nameSuccess && (
            <div className="p-3 bg-emerald-900/20 border border-emerald-500/50 rounded-lg text-emerald-400 text-sm">
              Name updated successfully!
            </div>
          )}

          <Button type="submit" variant="primary" isLoading={isUpdatingName} disabled={isUpdatingName || name === doctor?.name}>
            Update Name
          </Button>
        </form>
      </Card>

      {/* Password Update */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Change Password</h3>
        {!isPasswordUpdateSupported && (
          <div className="mb-4 p-3 bg-zinc-900/60 border border-zinc-700 rounded-lg text-zinc-400 text-sm">
            Password updates are not available yet. This will be enabled once the backend endpoint is ready.
          </div>
        )}
        <form onSubmit={handlePasswordUpdate} className="space-y-4">
          <Input
            label="Current Password"
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            disabled={isUpdatingPassword || !isPasswordUpdateSupported}
            autoComplete="current-password"
          />

          <Input
            label="New Password"
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            disabled={isUpdatingPassword || !isPasswordUpdateSupported}
            autoComplete="new-password"
            helperText="At least 6 characters"
          />

          <Input
            label="Confirm New Password"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            error={passwordError || undefined}
            disabled={isUpdatingPassword || !isPasswordUpdateSupported}
            autoComplete="new-password"
          />

          {passwordSuccess && (
            <div className="p-3 bg-emerald-900/20 border border-emerald-500/50 rounded-lg text-emerald-400 text-sm">
              Password updated successfully!
            </div>
          )}

          <Button
            type="submit"
            variant="primary"
            isLoading={isUpdatingPassword}
            disabled={
              !isPasswordUpdateSupported ||
              isUpdatingPassword ||
              !currentPassword ||
              !newPassword ||
              !confirmPassword
            }
          >
            Change Password
          </Button>
        </form>
      </Card>
    </div>
  );
}
