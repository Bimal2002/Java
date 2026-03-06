import java.util.Scanner;

public class weekday {
    public static void main(String [] args){
        Scanner scanner = new Scanner(System.in);
        int day = scanner.nextInt();
        switch (day){
            case 1:
                System.out.printf("Monday");
                break;
            case 2:
                System.out.printf("TuesDay");
                break;
            case 3:
                System.out.printf("Wen");
                break;
            case 4:
                System.out.printf("Thusday");
                break;
            case 5:
                System.out.printf("Friday");
                break;
            case 6:
                System.out.printf("Sat");
                break;
            case 7 :
                System.out.printf("Sun");
                break;
            default:
                System.out.printf("Invalid");
        }
    }
}
