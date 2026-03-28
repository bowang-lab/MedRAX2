/**
 * Validation Utilities
 *
 * Form validation functions.
 */

export const validation = {
  /**
   * Validate doctor name
   */
  doctorName: (name: string): string | null => {
    if (!name || name.trim().length === 0) {
      return "Name is required";
    }
    if (name.length < 2) {
      return "Name must be at least 2 characters";
    }
    if (name.length > 255) {
      return "Name must be less than 255 characters";
    }
    return null;
  },

  /**
   * Validate password
   */
  password: (password: string): string | null => {
    if (!password || password.length === 0) {
      return "Password is required";
    }
    if (password.length < 6) {
      return "Password must be at least 6 characters";
    }
    if (password.length > 100) {
      return "Password must be less than 100 characters";
    }
    return null;
  },

  /**
   * Validate patient name (optional)
   */
  patientName: (name: string | null): string | null => {
    // Patient name is optional, so null/empty is valid
    if (!name || name.trim().length === 0) {
      return null;
    }
    if (name.length > 255) {
      return "Name must be less than 255 characters";
    }
    return null;
  },

  /**
   * Validate chat name
   */
  chatName: (name: string): string | null => {
    if (!name || name.trim().length === 0) {
      return "Chat name is required";
    }
    if (name.length > 200) {
      return "Chat name must be less than 200 characters";
    }
    return null;
  },

  /**
   * Validate message content
   */
  messageContent: (content: string): string | null => {
    if (!content || content.trim().length === 0) {
      return "Message cannot be empty";
    }
    if (content.length > 10000) {
      return "Message must be less than 10000 characters";
    }
    return null;
  },

  /**
   * Validate question text
   */
  questionText: (text: string): string | null => {
    if (!text || text.trim().length === 0) {
      return "Question cannot be empty";
    }
    if (text.length > 500) {
      return "Question must be less than 500 characters";
    }
    return null;
  },

  /**
   * Validate file for upload
   */
  scanFile: (
    file: File,
    maxSizeBytes: number,
    allowedTypes: string[]
  ): string | null => {
    if (!file) {
      return "No file selected";
    }
    if (file.size > maxSizeBytes) {
      const maxSizeMB = maxSizeBytes / (1024 * 1024);
      return `File size must be less than ${maxSizeMB}MB`;
    }
    // Validate MIME type (can be spoofed, but provides UX feedback)
    if (!allowedTypes.includes(file.type)) {
      // Also check file extension as fallback
      const ext = file.name.toLowerCase().split(".").pop();
      const allowedExts = ["jpg", "jpeg", "png", "gif", "dcm", "dicom"];
      if (!ext || !allowedExts.includes(ext)) {
        return `File type ${file.type || "unknown"} is not allowed`;
      }
    }
    return null;
  },
};
