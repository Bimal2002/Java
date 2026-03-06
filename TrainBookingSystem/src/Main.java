import models.Booking;
import models.Train;
import models.User;
import repository.BookingRepository;
import repository.TrainRepository;
import repository.UserRepository;
import services.AuthService;
import services.BookingService;
import services.TrainService;
import utils.InputHelper;
import java.util.List;

public class Main {
    private final AuthService authService;
    private final TrainService trainService;
    private final BookingService bookingService;

    private Main() {
        UserRepository userRepository = new UserRepository();
        TrainRepository trainRepository = new TrainRepository();
        BookingRepository bookingRepository = new BookingRepository();

        this.authService = new AuthService(userRepository);
        this.trainService = new TrainService(trainRepository);
        this.bookingService = new BookingService(bookingRepository, trainRepository);
    }

    public static void main(String[] args) {
        new Main().start();
    }

    private void start() {
        while (true) {
            System.out.println("\n=== Train Booking System ===");
            System.out.println("1. Register");
            System.out.println("2. Login");
            System.out.println("3. Exit");

            int choice = InputHelper.readInt("Choose an option: ", 1, 3);
            switch (choice) {
                case 1 -> handleRegister();
                case 2 -> handleLogin();
                case 3 -> {
                    System.out.println("Exiting... Goodbye.");
                    return;
                }
                default -> System.out.println("Invalid option.");
            }
        }
    }

    private void handleRegister() {
        String name = InputHelper.readNonEmpty("Name: ");
        String email = InputHelper.readNonEmpty("Email: ");
        String password = InputHelper.readNonEmpty("Password: ");

        try {
            authService.register(name, email, password);
            System.out.println("Registration successful.");
        } catch (IllegalArgumentException e) {
            System.out.println("Registration failed: " + e.getMessage());
        }
    }

    private void handleLogin() {
        String email = InputHelper.readNonEmpty("Email: ");
        String password = InputHelper.readNonEmpty("Password: ");

        try {
            User user = authService.login(email, password);
            System.out.println("Login successful. Welcome, " + user.getName() + ".");
            userMenu(user);
        } catch (IllegalArgumentException e) {
            System.out.println("Login failed: " + e.getMessage());
        }
    }

    private void userMenu(User user) {
        while (true) {
            System.out.println("\n=== User Menu ===");
            System.out.println("1. List all trains");
            System.out.println("2. Search trains");
            System.out.println("3. Book ticket");
            System.out.println("4. View my bookings");
            System.out.println("5. Logout");

            int choice = InputHelper.readInt("Choose an option: ", 1, 5);
            switch (choice) {
                case 1 -> showTrains(trainService.getAllTrains());
                case 2 -> searchTrains();
                case 3 -> bookTicket(user);
                case 4 -> viewMyBookings(user);
                case 5 -> {
                    System.out.println("Logged out.");
                    return;
                }
                default -> System.out.println("Invalid option.");
            }
        }
    }

    private void searchTrains() {
        String source = InputHelper.readNonEmpty("Source city: ");
        String destination = InputHelper.readNonEmpty("Destination city: ");
        List<Train> results = trainService.searchTrains(source, destination);
        if (results.isEmpty()) {
            System.out.println("No trains found.");
            return;
        }
        showTrains(results);
    }

    private void bookTicket(User user) {
        String trainId = InputHelper.readNonEmpty("Train ID: ");
        String passengerName = InputHelper.readNonEmpty("Passenger name: ");
        int seats = InputHelper.readInt("Seats to book: ", 1, 10);

        try {
            Booking booking = bookingService.bookTicket(user, trainId, passengerName, seats);
            System.out.println("Booking successful.");
            System.out.println("Booking ID: " + booking.getId());
            System.out.println("Train: " + booking.getTrainName());
            System.out.println("Seats: " + booking.getSeatsBooked());
            System.out.println("Total Fare: " + booking.getTotalFare());
        } catch (IllegalArgumentException e) {
            System.out.println("Booking failed: " + e.getMessage());
        }
    }

    private void viewMyBookings(User user) {
        List<Booking> bookings = bookingService.getBookingsForUser(user.getId());
        if (bookings.isEmpty()) {
            System.out.println("No bookings found.");
            return;
        }

        System.out.println("\nMy Bookings:");
        for (Booking booking : bookings) {
            System.out.println("ID: " + booking.getId()
                    + " | Train: " + booking.getTrainName()
                    + " | Passenger: " + booking.getPassengerName()
                    + " | Seats: " + booking.getSeatsBooked()
                    + " | Fare: " + booking.getTotalFare()
                    + " | Time: " + booking.getBookingTime());
        }
    }

    private void showTrains(List<Train> trains) {
        System.out.println("\nAvailable Trains:");
        for (Train train : trains) {
            System.out.println("ID: " + train.getId()
                    + " | " + train.getName()
                    + " | " + train.getSource() + " -> " + train.getDestination()
                    + " | Seats: " + train.getAvailableSeats() + "/" + train.getTotalSeats()
                    + " | Fare: " + train.getFarePerSeat());
        }

    }
}
