package services;

import models.User;
import repository.UserRepository;
import utils.PasswordUtils;

import java.util.Optional;
import java.util.UUID;

public class AuthService {
    private final UserRepository userRepository;

    public AuthService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User register(String name, String email, String password) {
        Optional<User> existing = userRepository.findByEmail(email);
        if (existing.isPresent()) {
            throw new IllegalArgumentException("Email is already registered.");
        }

        User user = new User(
                UUID.randomUUID().toString(),
                name.trim(),
                email.trim(),
                PasswordUtils.hashPassword(password)
        );
        userRepository.save(user);
        return user;
    }

    public User login(String email, String password) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new IllegalArgumentException("User not found."));

        if (!PasswordUtils.verifyPassword(password, user.getPasswordHash())) {
            throw new IllegalArgumentException("Invalid password.");
        }
        return user;
    }

}
