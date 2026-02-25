/**
 * Doctor (User) Types
 *
 * Doctors are the authenticated users of the system.
 * Simple authentication with just name and password.
 */

export interface Doctor {
  id: string;
  name: string;
  createdAt: string;
}

export interface DoctorRegistration {
  name: string;
  password: string;
}

export interface DoctorLogin {
  name: string;
  password: string;
}

export interface AuthSession {
  token: string;
  doctor: Doctor;
  expiresAt: string;
}
