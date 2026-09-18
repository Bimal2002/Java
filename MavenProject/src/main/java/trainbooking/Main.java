package trainbooking;

import trainbooking.service.UserService;

import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws Exception {

        Scanner sc = new Scanner(System.in);

        UserService service = new UserService();

        System.out.println("Train Booking System");

        System.out.print("Enter Username: ");
        String username = sc.nextLine();

        System.out.print("Enter Password: ");
        String password = sc.nextLine();

        service.register(username, password);

    }

}