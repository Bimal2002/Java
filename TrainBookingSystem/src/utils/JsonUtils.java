package utils;

import models.User;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class JsonUtils {
    private static final String USER_FILE_PATH = "data/users.json";

    private JsonUtils() {
        // Utility class
    }

    public static List<User> readUsers() {
        try {
            File file = ensureFileExists(USER_FILE_PATH);
            List<String> lines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);
            List<User> users = new ArrayList<>();
            for (String line : lines) {
                if (line == null || line.trim().isEmpty()) {
                    continue;
                }
                String[] parts = splitEscaped(line, '|');
                if (parts.length != 4) {
                    continue;
                }
                users.add(new User(unescape(parts[0]), unescape(parts[1]), unescape(parts[2]), unescape(parts[3])));
            }
            return users;
        } catch (IOException e) {
            throw new RuntimeException("Failed to read users from file", e);
        }
    }

    public static void writeUsers(List<User> users) {
        try {
            File file = ensureFileExists(USER_FILE_PATH);
            List<String> lines = new ArrayList<>();
            for (User user : users) {
                String line = escape(user.getId()) + "|"
                        + escape(user.getName()) + "|"
                        + escape(user.getEmail()) + "|"
                        + escape(user.getPasswordHash());
                lines.add(line);
            }
            Files.write(file.toPath(), lines, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to write users to file", e);
        }
    }

    public static void appendUser(User user) {
        List<User> users = readUsers();
        users.add(user);
        writeUsers(users);
    }

    private static File ensureFileExists(String path) throws IOException {
        File file = new File(path);
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        if (!file.exists()) {
            file.createNewFile();
        }
        return file;
    }

    private static String escape(String input) {
        if (input == null) {
            return "";
        }
        return input.replace("\\", "\\\\").replace("|", "\\|");
    }

    private static String unescape(String input) {
        StringBuilder sb = new StringBuilder();
        boolean escaping = false;
        for (int i = 0; i < input.length(); i++) {
            char c = input.charAt(i);
            if (escaping) {
                sb.append(c);
                escaping = false;
            } else if (c == '\\') {
                escaping = true;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static String[] splitEscaped(String input, char delimiter) {
        List<String> parts = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean escaping = false;

        for (int i = 0; i < input.length(); i++) {
            char c = input.charAt(i);
            if (escaping) {
                current.append(c);
                escaping = false;
            } else if (c == '\\') {
                escaping = true;
            } else if (c == delimiter) {
                parts.add(current.toString());
                current.setLength(0);
            } else {
                current.append(c);
            }
        }
        parts.add(current.toString());
        return parts.toArray(new String[0]);
    }
}
