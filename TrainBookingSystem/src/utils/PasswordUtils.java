package utils;

import java.security.SecureRandom;
import java.util.Base64;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;

@SuppressWarnings("unused")
public class PasswordUtils {
    private static final SecureRandom RANDOM = new SecureRandom();
    private static final int SALT_SIZE = 16;
    private static final int ITERATIONS = 120_000;
    private static final int KEY_LENGTH = 256;

    private PasswordUtils() {
        // Utility class
    }

    @SuppressWarnings("unused")
    public static String hashPassword(String password){
        byte[] salt = new byte[SALT_SIZE];
        RANDOM.nextBytes(salt);
        byte[] hash = pbkdf2(password.toCharArray(), salt, ITERATIONS, KEY_LENGTH);
        return Base64.getEncoder().encodeToString(salt)
                + ":" + ITERATIONS
                + ":" + Base64.getEncoder().encodeToString(hash);
    }

    @SuppressWarnings("unused")
    public static boolean verifyPassword(String password,String hash){
        if (hash == null) {
            return false;
        }
        String[] parts = hash.split(":", 3);
        if (parts.length != 3) {
            return false;
        }

        byte[] salt;
        byte[] storedHash;
        int iterations;
        try {
            salt = Base64.getDecoder().decode(parts[0]);
            iterations = Integer.parseInt(parts[1]);
            storedHash = Base64.getDecoder().decode(parts[2]);
        } catch (IllegalArgumentException e) {
            return false;
        }

        byte[] computed = pbkdf2(password.toCharArray(), salt, iterations, storedHash.length * 8);
        return constantTimeEquals(storedHash, computed);
    }

    private static byte[] pbkdf2(char[] password, byte[] salt, int iterations, int keyLength) {
        try {
            PBEKeySpec spec = new PBEKeySpec(password, salt, iterations, keyLength);
            SecretKeyFactory skf = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256");
            return skf.generateSecret(spec).getEncoded();
        } catch (Exception e) {
            throw new IllegalStateException("Password hashing failed", e);
        }
    }

    private static boolean constantTimeEquals(byte[] a, byte[] b) {
        if (a == null || b == null || a.length != b.length) {
            return false;
        }
        int result = 0;
        for (int i = 0; i < a.length; i++) {
            result |= a[i] ^ b[i];
        }
        return result == 0;
    }
}
