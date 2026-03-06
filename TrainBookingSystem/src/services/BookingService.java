package services;

import models.Booking;
import models.Train;
import models.User;
import repository.BookingRepository;
import repository.TrainRepository;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.UUID;

public class BookingService {
    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final BookingRepository bookingRepository;
    private final TrainRepository trainRepository;

    public BookingService(BookingRepository bookingRepository, TrainRepository trainRepository) {
        this.bookingRepository = bookingRepository;
        this.trainRepository = trainRepository;
    }

    public Booking bookTicket(User user, String trainId, String passengerName, int seats) {
        if (seats <= 0) {
            throw new IllegalArgumentException("Seats should be at least 1.");
        }

        Train train = trainRepository.findById(trainId)
                .orElseThrow(() -> new IllegalArgumentException("Train not found for ID: " + trainId));

        if (train.getAvailableSeats() < seats) {
            throw new IllegalArgumentException("Not enough seats available.");
        }

        train.setAvailableSeats(train.getAvailableSeats() - seats);

        Booking booking = new Booking(
                UUID.randomUUID().toString(),
                user.getId(),
                train.getId(),
                train.getName(),
                passengerName,
                seats,
                seats * train.getFarePerSeat(),
                LocalDateTime.now().format(FORMATTER)
        );
        bookingRepository.save(booking);
        return booking;
    }

    public List<Booking> getBookingsForUser(String userId) {
        return bookingRepository.findByUserId(userId);
    }

}
