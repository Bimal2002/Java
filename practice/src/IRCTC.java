import java.util.*;
class Train{
    int trainId;
    String trainName;
    String source;
    String destination;
    int totalSeats;
    int avilableSeats;
    Train(int trainId,String trainName,String source,String destination,int totalSeats){
        this.trainId = trainId;
        this.trainName = trainName;
        this.source = source;
        this.destination = destination;
        this.totalSeats = totalSeats;
        this.avilableSeats = totalSeats;
    }
    void display(){
        System.out.println(trainId + " | " + trainName + " | "+ source + " -> "+ destination + " |Seats : "+ avilableSeats);
    }
}
class Ticket{
    int ticketId;
    Train train;
    String passenderName;
    int seatsBooked;

    Ticket(int ticketId,Train train,String passenderName,int seatsBooked){
        this.ticketId = ticketId;
        this.train = train;
        this.passenderName = passenderName;
        this.seatsBooked = seatsBooked;
    }
    void display(){
        System.out.println("TicketId : "+ ticketId);
        System.out.println("Passenger : "+ passenderName);
        System.out.println("Train : "+ train.trainName);
        System.out.println("Seats: "+seatsBooked);

    }
}
class IRCTCSystem{
    List<Train>trains = new ArrayList<>();
    Map<Integer,Ticket>bookigs = new HashMap<>();
    int ticketCounter = 1;
    void addTrain(Train train){
        trains.add(train);
    }
    void viewTrains(){
        for(Train t: trains){
            t.display();
        }
    }

    void searchTrain(String source,String destination){
        for(Train t : trains){
            if(t.source.equalsIgnoreCase(source) && t.destination.equalsIgnoreCase(destination)){
                t.display();
            }
        }
    }

    void bookTicket(int trainId,String name ,int seats){
        for(Train t : trains){
            if(t.trainId == trainId){
                if(t.avilableSeats >= seats){
                    t.avilableSeats -=seats;
                    Ticket ticket = new Ticket(ticketCounter,t,name,seats);
                    bookigs.put(ticketCounter,ticket);
                    System.out.println("Booking Successful!");
                    ticket.display();
                    ticketCounter++;
                }else{
                    System.out.println("Not enough seats! ");
                }
            }
        }
    }
    void cancelTicket(int ticketId) {

        if(bookigs.containsKey(ticketId)) {

            Ticket ticket = bookigs.get(ticketId);

            ticket.train.avilableSeats += ticket.seatsBooked;

            bookigs.remove(ticketId);

            System.out.println("Ticket Cancelled!");

        } else {
            System.out.println("Invalid Ticket ID");
        }
    }
    void viewBookings() {
        for(Ticket t : bookigs.values()) {
            t.display();
        }
    }

}
public class IRCTC {
    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);
        IRCTCSystem system = new IRCTCSystem();

        system.addTrain(new Train(1,"Express",
                "Kolkata","Delhi",100));
        system.addTrain(new Train(2,"Superfast",
                "Mumbai","Delhi",50));

        while(true) {

            System.out.println("\n1. View Trains");
            System.out.println("2. Search Train");
            System.out.println("3. Book Ticket");
            System.out.println("4. Cancel Ticket");
            System.out.println("5. View Bookings");
            System.out.println("6. Exit");

            int choice = sc.nextInt();

            switch(choice) {

                case 1:
                    system.viewTrains();
                    break;

                case 2:
                    sc.nextLine();
                    System.out.print("Source: ");
                    String source = sc.nextLine();
                    System.out.print("Destination: ");
                    String dest = sc.nextLine();
                    system.searchTrain(source, dest);
                    break;

                case 3:
                    System.out.print("Train ID: ");
                    int id = sc.nextInt();
                    sc.nextLine();
                    System.out.print("Passenger Name: ");
                    String name = sc.nextLine();
                    System.out.print("Seats: ");
                    int seats = sc.nextInt();
                    system.bookTicket(id, name, seats);
                    break;

                case 4:
                    System.out.print("Ticket ID: ");
                    int ticketId = sc.nextInt();
                    system.cancelTicket(ticketId);
                    break;

                case 5:
                    system.viewBookings();
                    break;

                case 6:
                    System.exit(0);
            }
        }
    }
}
